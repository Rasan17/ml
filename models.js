/**
 * Universal Medical ML/NeuralNet Predictor - machine learning and neural network (MLP) engine
 * Provides neural networks (multilayer perceptrons, TensorFlow.js + pure JS fallback), Random Forests,
 * Logistic/Softmax Regression, Ridge Regression, k-NN,
 * and Gaussian Naive Bayes with comprehensive evaluation metrics.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    /**
     * Pseudo-Random Number Generator (Mulberry32) for deterministic reproducible modeling
     */
    function createPRNG(seed) {
        if (seed === null || seed === undefined || seed === '' || isNaN(seed)) {
            return Math.random;
        }
        let s = Math.floor(Math.abs(Number(seed))) >>> 0;
        if (s === 0) s = 1;
        return function () {
            s |= 0; s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    // Standard normal CDF (Abramowitz & Stegun 7.1.26 via erf; |error| < 1.5e-7)
    function normalCdf(z) {
        const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
        const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(z * z) / 2);
        return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2;
    }

    // Log-gamma (Lanczos) and the regularised incomplete beta function (continued fraction, Numerical Recipes)
    function logGamma(x) {
        const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
        let y = x, tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        let ser = 1.000000000190015;
        for (let j = 0; j < 6; j++) ser += c[j] / ++y;
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }
    function betacf(a, b, x) {
        const MAXIT = 300, EPS = 3e-16, FPMIN = 1e-300;
        const qab = a + b, qap = a + 1, qam = a - 1;
        let c = 1, d = 1 - qab * x / qap;
        if (Math.abs(d) < FPMIN) d = FPMIN;
        d = 1 / d;
        let h = d;
        for (let m = 1; m <= MAXIT; m++) {
            const m2 = 2 * m;
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
            c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
            d = 1 / d; h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
            c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
            d = 1 / d;
            const del = d * c;
            h *= del;
            if (Math.abs(del - 1) < EPS) break;
        }
        return h;
    }
    function incompleteBeta(a, b, x) {
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
        return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
    }
    // Student t distribution: CDF and quantile (bisection on the CDF); infinite df uses the normal
    function tCdf(t, df) {
        if (!isFinite(df) || df > 1e7) return normalCdf(t);
        const x = df / (df + t * t);
        const tail = 0.5 * incompleteBeta(df / 2, 0.5, x);
        return t >= 0 ? 1 - tail : tail;
    }
    function tQuantile(p, df) {
        let lo = -1e3, hi = 1e3;
        for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (tCdf(mid, df) < p) lo = mid; else hi = mid; }
        return (lo + hi) / 2;
    }

    // Gaussian elimination with partial pivoting; returns null if singular
    function solveLinear(A, b) {
        const n = A.length;
        const M = A.map((r, i) => [...r, b[i]]);
        for (let c = 0; c < n; c++) {
            let piv = c;
            for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
            if (Math.abs(M[piv][c]) < 1e-12) return null;
            [M[c], M[piv]] = [M[piv], M[c]];
            for (let r = 0; r < n; r++) {
                if (r === c) continue;
                const f = M[r][c] / M[c][c];
                if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
            }
        }
        return M.map((r, i) => r[n] / r[i]);
    }

    function invertMatrix(A) {
        const n = A.length;
        const inv = [];
        for (let j = 0; j < n; j++) {
            const e = new Array(n).fill(0); e[j] = 1;
            const col = solveLinear(A, e);
            if (!col) return null;
            inv.push(col);
        }
        return inv[0].map((_, i) => inv.map(r => r[i])); // columns -> rows
    }

    /**
     * Yields control back to the browser between training epochs. TF.js's default yielding uses
     * requestAnimationFrame, which never fires in a background tab (training would freeze); here a
     * MessageChannel is used when the page is hidden, and setTimeout (allowing UI repaint) when visible.
     */
    function yieldToBrowser() {
        if (typeof document !== 'undefined' && document.hidden && typeof MessageChannel !== 'undefined') {
            return new Promise(resolve => {
                const ch = new MessageChannel();
                ch.port1.onmessage = () => resolve();
                ch.port2.postMessage(null);
            });
        }
        return new Promise(resolve => setTimeout(resolve, 0));
    }

    // Upper bound on mini-batch steps per neural-network epoch (see DeepNeuralNetwork.fit)
    const MAX_BATCHES_PER_EPOCH = 150;

    // Row-by-row outputs for any model, using a batched path when the model provides one
    function batchOutputs(model, X) {
        return typeof model.predictOutputs === 'function' ? model.predictOutputs(X) : X.map(x => model.predictSingle(x));
    }

    // Asynchronous variant: models that can yield to the browser while predicting (k-NN, ensembles
    // containing k-NN) do so; everything else falls back to the batched synchronous path
    async function batchOutputsAsync(model, X, opts = {}) {
        if (typeof model.predictOutputsAsync === 'function') return model.predictOutputsAsync(X, opts);
        const out = batchOutputs(model, X);
        await yieldToBrowser();
        if (typeof opts.onYield === 'function') opts.onYield(X.length, X.length);
        return out;
    }

    /**
     * Drives a step generator (one step = one tree / one epoch) asynchronously, yielding to the browser
     * every `every` steps so the UI can repaint and Force Stop can take effect. `opts.onYield` is called
     * at each yield; if it throws (e.g. user cancellation), training stops and the error propagates.
     */
    async function runStepsAsync(steps, opts = {}, every = 1) {
        let n = 0;
        for (const _ of steps) { // eslint-disable-line no-unused-vars
            if (++n % every === 0) {
                await yieldToBrowser();
                if (typeof opts.onYield === 'function') opts.onYield(n);
            }
        }
    }

    // ==========================================
    // 1. PURE JS DECISION TREE & RANDOM FOREST
    // ==========================================
    class DecisionTreeNode {
        constructor(isLeaf = false, value = null, featureIdx = null, threshold = null, left = null, right = null) {
            this.isLeaf = isLeaf;
            this.value = value; // For classification: class or prob dict; for regression: mean value
            this.featureIdx = featureIdx;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }
    }

    class DecisionTree {
        constructor(isClassification = true, maxDepth = 6, minSamplesSplit = 2, maxFeatures = null, rng = Math.random, exactSplits = false) {
            this.isClassification = isClassification;
            this.exactSplits = exactSplits; // true: every threshold between distinct values; false: ~10 per feature
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
            this.rng = rng || Math.random;
            this.root = null;
            this.featureImportances = {};
        }

        fit(X, y) {
            const numFeatures = X[0].length;
            this.featureImportances = new Array(numFeatures).fill(0);
            this.root = this._buildTree(X, y, 0);
            // Normalize feature importances
            const sum = this.featureImportances.reduce((a, b) => a + b, 0);
            if (sum > 0) {
                this.featureImportances = this.featureImportances.map(v => v / sum);
            }
        }

        _buildTree(X, y, depth) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            if (this.isClassification) {
                const uniqueClasses = [...new Set(y)];
                if (uniqueClasses.length === 1 || depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
                    const classCounts = {};
                    y.forEach(c => classCounts[c] = (classCounts[c] || 0) + 1);
                    let majority = y[0];
                    let maxCount = 0;
                    Object.keys(classCounts).forEach(c => {
                        if (classCounts[c] > maxCount) {
                            maxCount = classCounts[c];
                            majority = Number(c);
                        }
                    });
                    return new DecisionTreeNode(true, majority);
                }
            } else {
                const mean = y.reduce((a, b) => a + b, 0) / (numSamples || 1);
                if (depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
                    return new DecisionTreeNode(true, mean);
                }
            }

            // Choose random subset of features
            let featureIndices = Array.from({ length: numFeatures }, (_, i) => i);
            if (this.maxFeatures && this.maxFeatures < numFeatures) {
                // Seeded Fisher-Yates shuffle (a sort() with a random comparator is not a uniform shuffle)
                for (let i = featureIndices.length - 1; i > 0; i--) {
                    const j = Math.floor(this.rng() * (i + 1));
                    [featureIndices[i], featureIndices[j]] = [featureIndices[j], featureIndices[i]];
                }
                featureIndices = featureIndices.slice(0, this.maxFeatures);
            }

            let bestGain = -Infinity;
            let bestFeature = null;
            let bestThreshold = null;
            let bestSplits = null;

            const currentScore = this.isClassification ? this._gini(y) : this._variance(y);

            if (this.exactSplits) {
                // Exact search: sort once per feature and sweep left-to-right, scoring every threshold between
                // distinct values (O(n log n) per feature); the same gain criterion as the approximate search
                featureIndices.forEach(featIdx => {
                    const r = this._bestSplitExact(X, y, featIdx, currentScore);
                    if (r && r.gain > bestGain) { bestGain = r.gain; bestFeature = featIdx; bestThreshold = r.threshold; }
                });
                if (bestFeature !== null) {
                    const leftX = [], leftY = [], rightX = [], rightY = [];
                    for (let s = 0; s < numSamples; s++) {
                        if (X[s][bestFeature] <= bestThreshold) { leftX.push(X[s]); leftY.push(y[s]); } else { rightX.push(X[s]); rightY.push(y[s]); }
                    }
                    if (leftY.length && rightY.length) bestSplits = { leftX, leftY, rightX, rightY };
                }
            } else featureIndices.forEach(featIdx => {
                const values = X.map(r => r[featIdx]);
                // Sample potential split thresholds
                const uniqueVals = [...new Set(values)].sort((a, b) => a - b);
                if (uniqueVals.length <= 1) return;

                const step = Math.max(1, Math.floor(uniqueVals.length / 10));
                for (let i = 0; i < uniqueVals.length - 1; i += step) {
                    const threshold = (uniqueVals[i] + uniqueVals[i + 1]) / 2;
                    const leftX = [], leftY = [], rightX = [], rightY = [];

                    for (let s = 0; s < numSamples; s++) {
                        if (X[s][featIdx] <= threshold) {
                            leftX.push(X[s]);
                            leftY.push(y[s]);
                        } else {
                            rightX.push(X[s]);
                            rightY.push(y[s]);
                        }
                    }

                    if (leftY.length === 0 || rightY.length === 0) continue;

                    let gain = 0;
                    if (this.isClassification) {
                        const leftGini = this._gini(leftY);
                        const rightGini = this._gini(rightY);
                        gain = currentScore - ((leftY.length / numSamples) * leftGini + (rightY.length / numSamples) * rightGini);
                    } else {
                        const leftVar = this._variance(leftY);
                        const rightVar = this._variance(rightY);
                        gain = currentScore - ((leftY.length / numSamples) * leftVar + (rightY.length / numSamples) * rightVar);
                    }

                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = featIdx;
                        bestThreshold = threshold;
                        bestSplits = { leftX, leftY, rightX, rightY };
                    }
                }
            });

            if (bestGain <= 0 || !bestSplits) {
                if (this.isClassification) {
                    const counts = {};
                    y.forEach(c => counts[c] = (counts[c] || 0) + 1);
                    let majority = y[0];
                    let maxCount = 0;
                    Object.keys(counts).forEach(c => {
                        if (counts[c] > maxCount) {
                            maxCount = counts[c];
                            majority = Number(c);
                        }
                    });
                    return new DecisionTreeNode(true, majority);
                } else {
                    return new DecisionTreeNode(true, y.reduce((a, b) => a + b, 0) / (numSamples || 1));
                }
            }

            this.featureImportances[bestFeature] += bestGain * numSamples;

            const leftChild = this._buildTree(bestSplits.leftX, bestSplits.leftY, depth + 1);
            const rightChild = this._buildTree(bestSplits.rightX, bestSplits.rightY, depth + 1);

            return new DecisionTreeNode(false, null, bestFeature, bestThreshold, leftChild, rightChild);
        }

        _gini(y) {
            const counts = {};
            y.forEach(c => counts[c] = (counts[c] || 0) + 1);
            let sumSq = 0;
            const n = y.length;
            Object.values(counts).forEach(cnt => {
                const p = cnt / n;
                sumSq += p * p;
            });
            return 1 - sumSq;
        }

        _bestSplitExact(X, y, featIdx, currentScore) {
            const n = X.length;
            const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => X[a][featIdx] - X[b][featIdx]);
            if (X[order[0]][featIdx] === X[order[n - 1]][featIdx]) return null;
            let best = null;
            if (this.isClassification) {
                let k = 0;
                for (let i = 0; i < n; i++) if (y[i] + 1 > k) k = y[i] + 1;
                const right = new Float64Array(k), left = new Float64Array(k);
                for (let i = 0; i < n; i++) right[y[i]]++;
                let sqL = 0, sqR = 0;
                for (let c = 0; c < k; c++) sqR += right[c] * right[c];
                for (let t = 0; t < n - 1; t++) {
                    const c = y[order[t]];
                    // move one case from right to left, updating the sums of squared counts
                    sqR += -2 * right[c] + 1; right[c]--;
                    sqL += 2 * left[c] + 1; left[c]++;
                    const v = X[order[t]][featIdx], vNext = X[order[t + 1]][featIdx];
                    if (v === vNext) continue;
                    const nL = t + 1, nR = n - nL;
                    const giniL = 1 - sqL / (nL * nL), giniR = 1 - sqR / (nR * nR);
                    const gain = currentScore - ((nL / n) * giniL + (nR / n) * giniR);
                    if (!best || gain > best.gain) best = { gain, threshold: (v + vNext) / 2 };
                }
            } else {
                let sR = 0, qR = 0, sL = 0, qL = 0;
                for (let i = 0; i < n; i++) { sR += y[i]; qR += y[i] * y[i]; }
                for (let t = 0; t < n - 1; t++) {
                    const yv = y[order[t]];
                    sR -= yv; qR -= yv * yv; sL += yv; qL += yv * yv;
                    const v = X[order[t]][featIdx], vNext = X[order[t + 1]][featIdx];
                    if (v === vNext) continue;
                    const nL = t + 1, nR = n - nL;
                    const varL = Math.max(0, qL / nL - (sL / nL) ** 2), varR = Math.max(0, qR / nR - (sR / nR) ** 2);
                    const gain = currentScore - ((nL / n) * varL + (nR / n) * varR);
                    if (!best || gain > best.gain) best = { gain, threshold: (v + vNext) / 2 };
                }
            }
            return best;
        }

        _variance(y) {
            const n = y.length;
            if (n === 0) return 0;
            const mean = y.reduce((a, b) => a + b, 0) / n;
            return y.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
        }

        predictSample(x) {
            let node = this.root;
            while (!node.isLeaf) {
                if (x[node.featureIdx] <= node.threshold) {
                    node = node.left;
                } else {
                    node = node.right;
                }
            }
            return node.value;
        }
    }

    class RandomForest {
        constructor(isClassification = true, numTrees = 25, maxDepth = 5, minSamplesSplit = 2, featureRatio = 'sqrt', seed = null, splitMode = 'auto') {
            this.isClassification = isClassification;
            this.splitMode = splitMode; // 'auto' (exact below 5,000 cases), 'exact' or 'approx'
            this.numTrees = numTrees;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.featureRatio = featureRatio;
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.rng = createPRNG(this.seed);
            this.trees = [];
            this.featureImportances = [];
        }

        fit(X, y) {
            for (const _ of this._fitSteps(X, y)) { /* synchronous: run every tree */ } // eslint-disable-line no-unused-vars
        }

        // Same fit, but yields to the browser after every tree (keeps the UI and Force Stop responsive)
        async fitAsync(X, y, opts = {}) {
            await runStepsAsync(this._fitSteps(X, y), opts, 1);
        }

        *_fitSteps(X, y) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            let maxFeatures = Math.max(1, Math.floor(this.isClassification ? Math.sqrt(numFeatures) : numFeatures / 3));
            if (this.featureRatio === 'all') {
                maxFeatures = numFeatures;
            } else if (this.featureRatio === 'log2') {
                maxFeatures = Math.max(1, Math.floor(Math.log2(numFeatures)));
            } else if (this.featureRatio === '0.5') {
                maxFeatures = Math.max(1, Math.floor(numFeatures * 0.5));
            } else if (this.featureRatio === '0.75') {
                maxFeatures = Math.max(1, Math.floor(numFeatures * 0.75));
            }

            this.trees = [];
            const aggregatedImportances = new Array(numFeatures).fill(0);

            for (let t = 0; t < this.numTrees; t++) {
                // Bootstrap sample with replacement using seeded PRNG
                const sampleX = [];
                const sampleY = [];
                for (let i = 0; i < numSamples; i++) {
                    const randIdx = Math.floor(this.rng() * numSamples);
                    sampleX.push(X[randIdx]);
                    sampleY.push(y[randIdx]);
                }

                const exact = this.splitMode === 'exact' || (this.splitMode !== 'approx' && numSamples < 5000);
                const tree = new DecisionTree(this.isClassification, this.maxDepth, this.minSamplesSplit, maxFeatures, this.rng, exact);
                tree.fit(sampleX, sampleY);
                this.trees.push(tree);

                tree.featureImportances.forEach((imp, fIdx) => {
                    aggregatedImportances[fIdx] += imp;
                });
                yield t;
            }

            const total = aggregatedImportances.reduce((a, b) => a + b, 0);
            this.featureImportances = aggregatedImportances.map(v => total > 0 ? Number((v / total).toFixed(4)) : 0);
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }

        predictSingle(x) {
            const votes = this.trees.map(tree => tree.predictSample(x));
            if (this.isClassification) {
                const counts = {};
                votes.forEach(v => counts[v] = (counts[v] || 0) + 1);
                let majority = votes[0];
                let maxCount = 0;
                Object.keys(counts).forEach(v => {
                    if (counts[v] > maxCount) {
                        maxCount = counts[v];
                        majority = Number(v);
                    }
                });
                const probs = {};
                Object.keys(counts).forEach(k => probs[k] = counts[k] / votes.length);
                return { pred: majority, probabilities: probs, confidence: maxCount / votes.length, probabilityType: 'vote' };
            } else {
                const mean = votes.reduce((a, b) => a + b, 0) / votes.length;
                return { pred: Number(mean.toFixed(4)), confidence: null };
            }
        }
    }

    // ==========================================
    // 2. LOGISTIC / SOFTMAX & RIDGE REGRESSION
    // ==========================================
    class GeneralizedLinearModel {
        constructor(isClassification = true, numClasses = 2, learningRate = 0.05, epochs = 150, lambdaL2 = 0.01, solver = 'momentum', seed = null) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.lr = learningRate;
            this.epochs = epochs;
            this.lambdaL2 = lambdaL2;
            this.solver = solver;
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.rng = createPRNG(this.seed);
            this.weights = null; // matrix or vector
            this.bias = null;
        }

        fit(X, y) {
            for (const _ of this._fitSteps(X, y)) { /* synchronous: run every epoch */ } // eslint-disable-line no-unused-vars
        }

        // Same fit, but yields to the browser every 10 epochs (keeps the UI and Force Stop responsive)
        async fitAsync(X, y, opts = {}) {
            await runStepsAsync(this._fitSteps(X, y), opts, 10);
        }

        *_fitSteps(X, y) {
            const n = X.length;
            const p = X[0].length;
            const momentumBeta = this.solver === 'momentum' ? 0.85 : 0.0;

            if (this.isClassification) {
                // Multinomial Softmax or Binary Sigmoid
                this.weights = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                this.bias = new Array(this.numClasses).fill(0);
                const velW = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                const velB = new Array(this.numClasses).fill(0);

                for (let epoch = 0; epoch < this.epochs; epoch++) {
                    const gradW = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                    const gradB = new Array(this.numClasses).fill(0);

                    for (let i = 0; i < n; i++) {
                        const logits = new Array(this.numClasses).fill(0);
                        for (let c = 0; c < this.numClasses; c++) {
                            let dot = this.bias[c];
                            for (let j = 0; j < p; j++) {
                                dot += this.weights[c][j] * X[i][j];
                            }
                            logits[c] = dot;
                        }

                        // Softmax
                        const maxLogit = Math.max(...logits);
                        const exp = logits.map(l => Math.exp(l - maxLogit));
                        const sumExp = exp.reduce((a, b) => a + b, 0);
                        const probs = exp.map(e => e / (sumExp || 1));

                        const trueClass = y[i];
                        for (let c = 0; c < this.numClasses; c++) {
                            const err = probs[c] - (c === trueClass ? 1 : 0);
                            gradB[c] += err;
                            for (let j = 0; j < p; j++) {
                                gradW[c][j] += err * X[i][j];
                            }
                        }
                    }

                    // Update with momentum and regularization
                    for (let c = 0; c < this.numClasses; c++) {
                        const gB = gradB[c] / n;
                        velB[c] = momentumBeta * velB[c] + this.lr * gB;
                        this.bias[c] -= velB[c];

                        for (let j = 0; j < p; j++) {
                            const reg = this.lambdaL2 * this.weights[c][j];
                            const gW = (gradW[c][j] / n) + reg;
                            velW[c][j] = momentumBeta * velW[c][j] + this.lr * gW;
                            this.weights[c][j] -= velW[c][j];
                        }
                    }
                    this._checkFinite(epoch);
                    yield epoch;
                }
            } else {
                // Ridge Linear Regression via Gradient Descent with momentum.
                // Objective: (1/2n)·Σ(pred − y)² + (λ/2)·‖w‖² (intercept unpenalised), whose gradient is
                // (1/n)·Σ err·x + λw, exactly as computed below. This is equivalent to scikit-learn
                // Ridge(alpha = n·λ) (checked in tests/reference_check.py); the learning rate applies to it.
                this.weights = new Array(p).fill(0);
                this.bias = 0;
                const velW = new Array(p).fill(0);
                let velB = 0;

                for (let epoch = 0; epoch < this.epochs; epoch++) {
                    let gradB = 0;
                    const gradW = new Array(p).fill(0);

                    for (let i = 0; i < n; i++) {
                        let pred = this.bias;
                        for (let j = 0; j < p; j++) {
                            pred += this.weights[j] * X[i][j];
                        }
                        const err = pred - y[i];
                        gradB += err;
                        for (let j = 0; j < p; j++) {
                            gradW[j] += err * X[i][j];
                        }
                    }

                    const gB = gradB / n;
                    velB = momentumBeta * velB + this.lr * gB;
                    this.bias -= velB;

                    for (let j = 0; j < p; j++) {
                        const reg = this.lambdaL2 * this.weights[j];
                        const gW = (gradW[j] / n) + reg;
                        velW[j] = momentumBeta * velW[j] + this.lr * gW;
                        this.weights[j] -= velW[j];
                    }
                    this._checkFinite(epoch);
                    yield epoch;
                }
            }
        }

        _checkFinite(epoch) {
            const flat = this.isClassification ? [...this.bias, ...this.weights.flat()] : [this.bias, ...this.weights];
            if (flat.some(v => !Number.isFinite(v))) {
                throw new Error(`Logistic/Ridge regression diverged at iteration ${epoch + 1} (coefficients became non-finite). Reduce the learning rate or increase the L2 penalty.`);
            }
        }

        predictSingle(x) {
            const p = x.length;
            if (this.isClassification) {
                const logits = new Array(this.numClasses).fill(0);
                for (let c = 0; c < this.numClasses; c++) {
                    let dot = this.bias[c];
                    for (let j = 0; j < p; j++) {
                        dot += this.weights[c][j] * x[j];
                    }
                    logits[c] = dot;
                }
                const maxLogit = Math.max(...logits);
                const exp = logits.map(l => Math.exp(l - maxLogit));
                const sumExp = exp.reduce((a, b) => a + b, 0);
                const probs = exp.map(e => e / (sumExp || 1));

                let bestClass = 0;
                let maxP = 0;
                const probMap = {};
                probs.forEach((prob, c) => {
                    probMap[c] = Number(prob.toFixed(4));
                    if (prob > maxP) {
                        maxP = prob;
                        bestClass = c;
                    }
                });

                return { pred: bestClass, probabilities: probMap, confidence: Number(maxP.toFixed(4)), probabilityType: 'model' };
            } else {
                let pred = this.bias;
                for (let j = 0; j < p; j++) {
                    pred += this.weights[j] * x[j];
                }
                return { pred: Number(pred.toFixed(4)), confidence: null };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 3. k-NEAREST NEIGHBORS (k-NN)
    // ==========================================
    class KNNModel {
        constructor(isClassification = true, k = 5, metric = 'euclidean', weighting = 'uniform', seed = null) {
            this.isClassification = isClassification;
            this.k = k;
            this.metric = metric; // 'euclidean', 'manhattan', 'chebyshev'
            this.weighting = weighting; // 'uniform', 'distance'
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.trainX = [];
            this.trainY = [];
        }

        fit(X, y) {
            this.trainX = X;
            this.trainY = y;
            // Seeded tie-break key per training case: equidistant neighbours are ordered by this key
            // (otherwise ties always favour rows earlier in the CSV); identical seed -> identical result
            const rng = createPRNG(this.seed);
            this.tieKeys = Float64Array.from(X, () => rng());
            // Flat typed-array copy of the training matrix: same values and arithmetic, much faster scans
            const n = X.length;
            const p = n ? X[0].length : 0;
            this.p = p;
            this.trainFlat = new Float64Array(n * p);
            for (let i = 0; i < n; i++) this.trainFlat.set(X[i], i * p);
        }

        predictSingle(x) {
            const n = this.trainY.length;
            const p = this.p;
            const T = this.trainFlat;
            const metric = this.metric;
            const tie = this.tieKeys;
            // Top-k selection (k is small) over (distance, seeded tie key) without materialising all
            // distances as objects: same neighbours and order as a full sort, at O(n·p + n·k)
            const kEff = Math.min(this.k, n);
            const bestD = new Float64Array(kEff);
            const bestI = new Int32Array(kEff);
            let count = 0;
            for (let i = 0; i < n; i++) {
                const off = i * p;
                let d = 0;
                if (metric === 'manhattan') {
                    for (let j = 0; j < p; j++) d += Math.abs(x[j] - T[off + j]);
                } else if (metric === 'chebyshev') {
                    for (let j = 0; j < p; j++) d = Math.max(d, Math.abs(x[j] - T[off + j]));
                } else {
                    let sumSq = 0;
                    for (let j = 0; j < p; j++) {
                        const diff = x[j] - T[off + j];
                        sumSq += diff * diff;
                    }
                    d = Math.sqrt(sumSq);
                }
                const ti = tie[i];
                if (count === kEff) {
                    const lastD = bestD[kEff - 1];
                    if (!(d < lastD || (d === lastD && ti < tie[bestI[kEff - 1]]))) continue;
                }
                let pos = count < kEff ? count : kEff - 1;
                while (pos > 0 && (d < bestD[pos - 1] || (d === bestD[pos - 1] && ti < tie[bestI[pos - 1]]))) {
                    bestD[pos] = bestD[pos - 1];
                    bestI[pos] = bestI[pos - 1];
                    pos--;
                }
                bestD[pos] = d;
                bestI[pos] = i;
                if (count < kEff) count++;
            }
            const kNearest = [];
            for (let r = 0; r < count; r++) kNearest.push({ dist: bestD[r], label: this.trainY[bestI[r]] });

            if (this.isClassification) {
                const weightedCounts = {};
                let totalWeight = 0;
                kNearest.forEach(n => {
                    const w = this.weighting === 'distance' ? (1.0 / (n.dist + 1e-5)) : 1.0;
                    weightedCounts[n.label] = (weightedCounts[n.label] || 0) + w;
                    totalWeight += w;
                });

                let majority = kNearest[0].label;
                let maxW = 0;
                Object.keys(weightedCounts).forEach(l => {
                    if (weightedCounts[l] > maxW) {
                        maxW = weightedCounts[l];
                        majority = Number(l);
                    }
                });

                const probs = {};
                Object.keys(weightedCounts).forEach(k => {
                    probs[k] = Number((weightedCounts[k] / (totalWeight || 1)).toFixed(4));
                });
                return {
                    pred: majority,
                    probabilities: probs,
                    confidence: Number((maxW / (totalWeight || 1)).toFixed(4)),
                    probabilityType: 'neighbours'
                };
            } else {
                let weightedSum = 0;
                let totalWeight = 0;
                kNearest.forEach(n => {
                    const w = this.weighting === 'distance' ? (1.0 / (n.dist + 1e-5)) : 1.0;
                    weightedSum += n.label * w;
                    totalWeight += w;
                });
                const mean = weightedSum / (totalWeight || 1);
                return { pred: Number(mean.toFixed(4)), confidence: null };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }

        predictOutputs(X) {
            return X.map(x => this.predictSingle(x));
        }

        // Same outputs, computed in chunks that yield to the browser (a large test set against a large
        // training set is billions of operations); opts.onYield(done, total) may throw to cancel
        async predictOutputsAsync(X, opts = {}) {
            const out = new Array(X.length);
            const perRow = Math.max(1, this.trainY.length * Math.max(1, this.p));
            const chunk = Math.max(1, Math.floor(6e7 / perRow)); // ~50 ms of work between yields
            for (let start = 0; start < X.length; start += chunk) {
                const end = Math.min(X.length, start + chunk);
                for (let r = start; r < end; r++) out[r] = this.predictSingle(X[r]);
                if (end < X.length) {
                    await yieldToBrowser();
                    if (typeof opts.onYield === 'function') opts.onYield(end, X.length);
                }
            }
            return out;
        }
    }

    // ==========================================
    // 4. NAIVE BAYES (mixed likelihoods)
    // Features that take only the values 0/1 in the training data (one-hot category indicators,
    // binary flags) use a Bernoulli likelihood with Laplace smoothing; all other features use a
    // Gaussian likelihood. (Class name kept for backwards compatibility.)
    // ==========================================
    class GaussianNaiveBayes {
        constructor(varSmoothing = 1e-4, priorMode = 'empirical', seed = null) {
            this.varSmoothing = varSmoothing; // 1e-9, 1e-4, 1e-2, 0.1
            this.priorMode = priorMode; // 'empirical', 'uniform', 'laplace'
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.classes = [];
            this.priors = {};
            this.means = {};
            this.variances = {};
            this.bernoulliP = {};
            this.isBinaryFeature = [];
        }

        fit(X, y) {
            this.classes = [...new Set(y)];
            const n = X.length;
            const p = X[0].length;
            const numK = this.classes.length;
            this.priors = {};
            this.means = {};
            this.variances = {};
            this.bernoulliP = {};
            this.isBinaryFeature = Array.from({ length: p }, (_, j) => X.every(r => r[j] === 0 || r[j] === 1));

            this.classes.forEach(c => {
                const classX = X.filter((_, idx) => y[idx] === c);
                const count = classX.length;

                if (this.priorMode === 'uniform') {
                    this.priors[c] = 1.0 / (numK || 1);
                } else if (this.priorMode === 'laplace') {
                    this.priors[c] = (count + 1) / (n + numK);
                } else {
                    // empirical
                    this.priors[c] = count / (n || 1);
                }

                this.means[c] = new Array(p).fill(0);
                this.variances[c] = new Array(p).fill(this.varSmoothing);
                this.bernoulliP[c] = new Array(p).fill(0.5);

                for (let j = 0; j < p; j++) {
                    const vals = classX.map(r => r[j]);
                    if (this.isBinaryFeature[j]) {
                        // Laplace-smoothed P(feature = 1 | class)
                        const ones = vals.reduce((a, b) => a + b, 0);
                        this.bernoulliP[c][j] = (ones + 1) / (vals.length + 2);
                        continue;
                    }
                    const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
                    this.means[c][j] = mean;
                    const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
                    this.variances[c][j] = Math.max(variance, this.varSmoothing);
                }
            });
        }

        predictSingle(x) {
            const logProbs = {};
            const p = x.length;

            this.classes.forEach(c => {
                let logP = Math.log(this.priors[c] || 1e-6);
                for (let j = 0; j < p; j++) {
                    if (this.isBinaryFeature[j]) {
                        const pj = this.bernoulliP[c][j];
                        logP += x[j] >= 0.5 ? Math.log(pj) : Math.log(1 - pj);
                        continue;
                    }
                    const mean = this.means[c][j];
                    const v = this.variances[c][j];
                    const diff = x[j] - mean;
                    logP += -0.5 * Math.log(2 * Math.PI * v) - (diff * diff) / (2 * v);
                }
                logProbs[c] = logP;
            });

            const maxLogP = Math.max(...Object.values(logProbs));
            const exps = {};
            let sumExp = 0;
            Object.keys(logProbs).forEach(c => {
                const exp = Math.exp(logProbs[c] - maxLogP);
                exps[c] = exp;
                sumExp += exp;
            });

            let bestClass = this.classes[0];
            let maxProb = 0;
            const probMap = {};
            Object.keys(exps).forEach(c => {
                const prob = exps[c] / (sumExp || 1);
                probMap[c] = Number(prob.toFixed(4));
                if (prob > maxProb) {
                    maxProb = prob;
                    bestClass = Number(c);
                }
            });

            return { pred: bestClass, probabilities: probMap, confidence: Number(maxProb.toFixed(4)), probabilityType: 'model' };
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 5. TENSORFLOW.JS NEURAL NETWORK (MULTILAYER PERCEPTRON)
    // ==========================================
    class DeepNeuralNetwork {
        constructor(isClassification = true, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.hiddenLayers = config.hiddenLayers || [64, 32];
            this.activation = config.activation || 'relu';
            this.learningRate = config.learningRate || 0.005;
            this.epochs = config.epochs || 50;
            this.batchSize = config.batchSize || 16;
            this.dropoutRate = config.dropoutRate !== undefined ? config.dropoutRate : 0.1;
            this.optimizerName = config.optimizer || 'adam';
            this.l2Reg = config.l2Reg !== undefined ? config.l2Reg : 0.0001;
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.rng = createPRNG(this.seed);
            this.tfModel = null;
            this.fallbackGLM = null;
            this.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
            // Internal validation split (carved from the TRAINING data) used for early stopping
            this.earlyStopping = config.earlyStopping !== false;
            this.validationFraction = config.validationFraction !== undefined ? config.validationFraction : 0.15;
            this.patience = config.patience || Math.max(5, Math.round(this.epochs * 0.2));
            this.trainingInfo = null;
        }

        /**
         * Splits training data into fit / validation parts (stratified by class for classification),
         * using the model's seeded RNG. Returns null when there are too few cases to hold any out.
         */
        _carveValidationSplit(X, Y) {
            const n = X.length;
            if (!this.earlyStopping || this.validationFraction <= 0 || n < 20) return null;
            const idx = Array.from({ length: n }, (_, i) => i);
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(this.rng() * (i + 1));
                [idx[i], idx[j]] = [idx[j], idx[i]];
            }
            const fitIdx = [];
            const valIdx = [];
            if (this.isClassification) {
                const byClass = {};
                idx.forEach(i => { (byClass[Y[i]] = byClass[Y[i]] || []).push(i); });
                Object.values(byClass).forEach(list => {
                    // Keep at least 2 cases of every class for fitting
                    const nVal = list.length >= 4 ? Math.max(1, Math.round(list.length * this.validationFraction)) : 0;
                    valIdx.push(...list.slice(0, nVal));
                    fitIdx.push(...list.slice(nVal));
                });
            } else {
                const nVal = Math.max(1, Math.round(n * this.validationFraction));
                valIdx.push(...idx.slice(0, nVal));
                fitIdx.push(...idx.slice(nVal));
            }
            if (valIdx.length < 3) return null;
            return {
                fitX: fitIdx.map(i => X[i]), fitY: fitIdx.map(i => Y[i]),
                valX: valIdx.map(i => X[i]), valY: valIdx.map(i => Y[i])
            };
        }

        /**
         * validX / validY: optional explicit validation data. It must come from the training data,
         * NEVER the test set. When omitted, an internal stratified validation split is carved from
         * trainX (see validationFraction) and used for early stopping with best-weight restoration.
         */
        async fit(trainX, trainY, validX = null, validY = null, onEpochCallback = null) {
            const hasTF = typeof window.tf !== 'undefined' && window.tf !== null;
            const inputDim = trainX[0].length;
            this.usedFallback = false;
            this.fallbackReason = null;
            this.trainingInfo = null;
            // Fresh seeded RNG per fit so refits with the same seed reproduce the same shuffles
            this.rng = createPRNG(this.seed);

            const allTrainX = trainX;
            const allTrainY = trainY;
            if (!(validX && validY && validX.length > 0)) {
                const carved = this._carveValidationSplit(trainX, trainY);
                if (carved) {
                    trainX = carved.fitX;
                    trainY = carved.fitY;
                    validX = carved.valX;
                    validY = carved.valY;
                } else {
                    validX = null;
                    validY = null;
                }
            }

            if (hasTF) {
                // Dispose existing model instance (and its optimizer state) if retraining
                this.dispose();

                let model = null;
                let valXTensor = null;
                let valYTensor = null;
                let xAll = null;
                let yAll = null;
                let bestWeights = null;
                const disposeBest = () => {
                    if (bestWeights) bestWeights.forEach(w => { try { w.dispose(); } catch (e) {} });
                    bestWeights = null;
                };

                try {
                    const tf = window.tf;
                    model = tf.sequential();

                    // Optional L2 kernel regularizer
                    const kernelReg = (this.l2Reg && this.l2Reg > 0 && typeof tf.regularizers !== 'undefined' && tf.regularizers.l2)
                        ? tf.regularizers.l2({ l2: this.l2Reg })
                        : null;

                    // Seed-based weight initializer for reproducibility (a different derived seed per layer)
                    const hasSeededInit = this.seed !== null && typeof tf.initializers !== 'undefined' && tf.initializers.glorotUniform;
                    const kernelInitFor = (layerIdx) => (hasSeededInit
                        ? tf.initializers.glorotUniform({ seed: this.seed + layerIdx * 7919 })
                        : 'glorotUniform');

                    // Input + First Hidden Layer
                    const firstDense = {
                        inputShape: [inputDim],
                        units: this.hiddenLayers[0] || 64,
                        activation: this.activation,
                        kernelInitializer: kernelInitFor(0)
                    };
                    if (kernelReg) firstDense.kernelRegularizer = kernelReg;
                    model.add(tf.layers.dense(firstDense));

                    if (this.dropoutRate > 0) {
                        const dropCfg = { rate: this.dropoutRate };
                        if (this.seed !== null) dropCfg.seed = this.seed;
                        model.add(tf.layers.dropout(dropCfg));
                    }

                    // Subsequent Hidden Layers
                    for (let l = 1; l < this.hiddenLayers.length; l++) {
                        const units = this.hiddenLayers[l];
                        if (!units || units <= 0) continue;
                        const nextDense = {
                            units: units,
                            activation: this.activation,
                            kernelInitializer: kernelInitFor(l)
                        };
                        if (kernelReg) nextDense.kernelRegularizer = kernelReg;
                        model.add(tf.layers.dense(nextDense));
                        if (this.dropoutRate > 0) {
                            const dropCfg = { rate: this.dropoutRate };
                            if (this.seed !== null) dropCfg.seed = this.seed + l;
                            model.add(tf.layers.dropout(dropCfg));
                        }
                    }

                    // Output Layer
                    const outIdx = this.hiddenLayers.length;
                    if (this.isClassification) {
                        model.add(tf.layers.dense({
                            units: this.numClasses,
                            activation: 'softmax',
                            kernelInitializer: kernelInitFor(outIdx)
                        }));
                    } else {
                        model.add(tf.layers.dense({ units: 1, kernelInitializer: kernelInitFor(outIdx) }));
                    }

                    // Optimizer
                    let opt;
                    if (this.optimizerName === 'sgd') {
                        opt = tf.train.sgd(this.learningRate);
                    } else if (this.optimizerName === 'rmsprop') {
                        opt = tf.train.rmsprop(this.learningRate);
                    } else {
                        opt = tf.train.adam(this.learningRate);
                    }

                    model.compile({
                        optimizer: opt,
                        loss: this.isClassification ? 'sparseCategoricalCrossentropy' : 'meanSquaredError',
                        metrics: this.isClassification ? ['accuracy'] : ['mse']
                    });

                    // Validation data (from the training data only) drives early stopping
                    let validationData = undefined;
                    if (validX && validY && validX.length > 0) {
                        valXTensor = tf.tensor2d(validX, [validX.length, inputDim]);
                        // Class indices as float32: TF.js 4.x's accuracy metric fails on int32 labels
                        valYTensor = this.isClassification ?
                            tf.tensor1d(validY, 'float32') :
                            tf.tensor2d(validY.map(v => [v]), [validY.length, 1]);
                        validationData = [valXTensor, valYTensor];
                    }

                    const history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
                    const n = trainX.length;
                    // Each mini-batch costs a fixed browser->GPU round trip, so on large cohorts a small batch
                    // (e.g. 16 -> thousands of steps per epoch) makes training take hours. The batch is raised
                    // just enough to keep about MAX_BATCHES_PER_EPOCH steps per epoch; small cohorts unchanged.
                    const batchSize = Math.min(n, Math.max(this.batchSize, Math.ceil(n / MAX_BATCHES_PER_EPOCH)));
                    let bestValLoss = Infinity;
                    let bestEpoch = 0;
                    let epochsWithoutImprovement = 0;
                    let stoppedEpoch = this.epochs;
                    let earlyStopped = false;

                    // Training tensors are built once; each epoch reorders them with tf.gather
                    xAll = tf.tensor2d(trainX, [n, inputDim]);
                    yAll = this.isClassification ?
                        tf.tensor1d(trainY, 'float32') :
                        tf.tensor2d(trainY.map(v => [v]), [n, 1]);

                    // Epoch loop with our own seeded shuffle (TF.js's built-in shuffle uses an unseeded RNG)
                    for (let epoch = 0; epoch < this.epochs; epoch++) {
                        const perm = Array.from({ length: n }, (_, i) => i);
                        for (let i = n - 1; i > 0; i--) {
                            const j = Math.floor(this.rng() * (i + 1));
                            [perm[i], perm[j]] = [perm[j], perm[i]];
                        }
                        const permTensor = tf.tensor1d(perm, 'int32');
                        const xTensor = tf.gather(xAll, permTensor);
                        const yTensor = tf.gather(yAll, permTensor);
                        permTensor.dispose();

                        let h;
                        try {
                            h = await model.fit(xTensor, yTensor, {
                                epochs: 1,
                                batchSize,
                                validationData,
                                shuffle: false,
                                yieldEvery: 'never' // we yield per epoch ourselves (works in background tabs)
                            });
                        } finally {
                            xTensor.dispose();
                            yTensor.dispose();
                        }

                        const H = h.history;
                        const logs = {
                            loss: H.loss ? H.loss[0] : NaN,
                            val_loss: H.val_loss ? H.val_loss[0] : undefined,
                            acc: H.acc ? H.acc[0] : (H.accuracy ? H.accuracy[0] : undefined),
                            val_acc: H.val_acc ? H.val_acc[0] : (H.val_accuracy ? H.val_accuracy[0] : undefined)
                        };
                        if (!Number.isFinite(logs.loss)) {
                            throw new Error(`Neural network training diverged at epoch ${epoch + 1} (loss is not finite). Reduce the learning rate.`);
                        }

                        history.loss.push(Number(logs.loss.toFixed(4)));
                        if (logs.val_loss !== undefined) history.valLoss.push(Number(logs.val_loss.toFixed(4)));
                        if (logs.acc !== undefined) history.accuracy.push(Number(logs.acc.toFixed(4)));
                        if (logs.val_acc !== undefined) history.valAccuracy.push(Number(logs.val_acc.toFixed(4)));

                        if (onEpochCallback) {
                            try {
                                onEpochCallback(epoch + 1, this.epochs, logs);
                            } catch (cbErr) {
                                // e.g. user cancellation via the watchdog: propagate, do not fall back
                                if (cbErr && typeof cbErr === 'object') cbErr.isCallbackError = true;
                                throw cbErr;
                            }
                        }
                        // Early stopping on internal validation loss, keeping the best weights
                        if (validationData && this.earlyStopping && Number.isFinite(logs.val_loss)) {
                            if (logs.val_loss < bestValLoss - 1e-6) {
                                bestValLoss = logs.val_loss;
                                bestEpoch = epoch + 1;
                                epochsWithoutImprovement = 0;
                                disposeBest();
                                bestWeights = model.getWeights().map(w => w.clone());
                            } else if (++epochsWithoutImprovement >= this.patience) {
                                stoppedEpoch = epoch + 1;
                                earlyStopped = true;
                                await yieldToBrowser();
                                break;
                            }
                        }
                        await yieldToBrowser();
                    }

                    if (bestWeights) {
                        model.setWeights(bestWeights);
                        disposeBest();
                    }
                    this.trainingInfo = {
                        fitCases: trainX.length,
                        validationCases: validationData ? validX.length : 0,
                        earlyStopping: Boolean(validationData && this.earlyStopping),
                        requestedBatchSize: this.batchSize,
                        batchSize,
                        earlyStopped,
                        bestEpoch: bestEpoch || null,
                        stoppedEpoch,
                        maxEpochs: this.epochs
                    };

                    this.tfModel = model;
                    model = null; // ownership transferred
                    this.history = history;
                    return;
                } catch (e) {
                    if (e && e.isCallbackError) throw e;
                    console.warn("TensorFlow.js training exception, using native fallback:", e);
                    this.fallbackReason = e && e.message ? e.message : String(e);
                } finally {
                    // Guaranteed cleanup of intermediate tensors and of a partially-built model
                    disposeBest();
                    if (xAll) { try { xAll.dispose(); } catch (e) {} }
                    if (yAll) { try { yAll.dispose(); } catch (e) {} }
                    if (model) {
                        const opt = model.optimizer;
                        try { model.dispose(); } catch (e) {}
                        try { if (opt && typeof opt.dispose === 'function') opt.dispose(); } catch (e) {}
                    }
                    if (valXTensor) { try { valXTensor.dispose(); } catch (e) {} }
                    if (valYTensor) { try { valYTensor.dispose(); } catch (e) {} }
                }
            } else {
                this.fallbackReason = 'TensorFlow.js is not available';
            }

            // Fallback: native (single-layer) logistic / softmax / ridge model.
            // This is NOT a neural network; callers can detect it via `usedFallback`.
            console.warn(`Neural network unavailable (${this.fallbackReason}); trained a linear fallback model instead.`);
            this.usedFallback = true;
            this.fallbackGLM = new GeneralizedLinearModel(
                this.isClassification,
                this.numClasses,
                0.05,
                Math.max(150, this.epochs),
                this.l2Reg,
                'momentum',
                this.seed
            );
            // The fallback has no early stopping, so it uses all of the training data
            this.fallbackGLM.fit(allTrainX, allTrainY);
            // No epoch-level training history exists for the fallback model
            this.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
        }

        predictSingle(x) {
            return this.predictOutputs([x])[0];
        }

        // Batched prediction: one tensor and one GPU->CPU read for all rows (per-row dataSync() calls
        // stall the browser on large evaluation or batch files)
        predictOutputs(X) {
            if (this.tfModel) {
                if (X.length === 0) return [];
                const tf = window.tf;
                const width = this.isClassification ? this.numClasses : 1;
                const data = tf.tidy(() => this.tfModel.predict(tf.tensor2d(X, [X.length, X[0].length])).dataSync());
                return X.map((_, r) => {
                    const row = data.subarray(r * width, (r + 1) * width);
                    if (!this.isClassification) {
                        return { pred: Number(row[0].toFixed(4)), confidence: null };
                    }
                    let bestClass = 0;
                    let maxP = 0;
                    const probMap = {};
                    for (let c = 0; c < this.numClasses; c++) {
                        const prob = row[c];
                        probMap[c] = Number(prob.toFixed(4));
                        if (prob > maxP) {
                            maxP = prob;
                            bestClass = c;
                        }
                    }
                    return { pred: bestClass, probabilities: probMap, confidence: Number(maxP.toFixed(4)), probabilityType: 'model' };
                });
            } else if (this.fallbackGLM) {
                return X.map(x => this.fallbackGLM.predictSingle(x));
            }
            throw new Error("Model is not trained yet.");
        }

        predict(X) {
            return this.predictOutputs(X).map(o => o.pred);
        }

        // Frees the TensorFlow.js (GPU/WebGL) memory held by this model. model.dispose() does NOT free the
        // optimizer's state (e.g. Adam's moment estimates), so the optimizer is disposed explicitly too.
        dispose() {
            if (this.tfModel) {
                const opt = this.tfModel.optimizer;
                try { this.tfModel.dispose(); } catch (e) { /* ignore */ }
                try { if (opt && typeof opt.dispose === 'function') opt.dispose(); } catch (e) { /* ignore */ }
                this.tfModel = null;
            }
        }
    }

    // ==========================================
    // 6. ENSEMBLE SUPER LEARNER (VOTING & BLENDING)
    // ==========================================
    class EnsembleModel {
        constructor(isClassification, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.voting = config.voting || 'soft'; // 'soft', 'weighted', 'hard'
            this.selectedBaseIds = config.baseModels && config.baseModels.length > 0
                ? config.baseModels
                : ['nn', 'rf', 'glm'];
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.rng = createPRNG(this.seed);
            this.models = [];
            this.weights = [];
            this.baseMetrics = [];
            this.userHyperparams = config;
        }

        async _trainBaseModels(trainX, trainY, onProgress, phaseLabel = '') {
            const models = [];
            const baseIds = this.selectedBaseIds.filter(mId => this.isClassification || mId !== 'nb'); // NB is classification only
            const totalBases = baseIds.length;
            const capacity = this.userHyperparams.capacity || 'balanced';

            for (let idx = 0; idx < totalBases; idx++) {
                const mId = baseIds[idx];

                if (onProgress) {
                    onProgress(idx + 1, totalBases, {
                        stage: `${phaseLabel}Training Ensemble Base Model ${idx + 1}/${totalBases}: ${mId.toUpperCase()}...`,
                        baseId: mId
                    });
                }

                const subSeed = this.seed !== null ? (this.seed + (idx + 1) * 1007) : null;
                let subModel = null;
                if (mId === 'nn') {
                    const epochs = capacity === 'fast' ? 20 : (capacity === 'deep' ? 60 : (this.userHyperparams.nnEpochs || 35));
                    const hidden = capacity === 'fast' ? [32, 16] : (capacity === 'deep' ? [64, 48, 24] : [48, 24]);
                    const nnParams = {
                        epochs: epochs,
                        learningRate: this.userHyperparams.nnLr || 0.005,
                        hiddenLayers: hidden,
                        dropoutRate: 0.1,
                        activation: 'relu',
                        optimizer: 'adam',
                        batchSize: 16,
                        l2Reg: 0.0001,
                        seed: subSeed
                    };
                    subModel = new DeepNeuralNetwork(this.isClassification, this.numClasses, nnParams);
                    await subModel.fit(trainX, trainY, null, null, (ep, tot, logs) => {
                        if (onProgress) {
                            onProgress(idx + 1, totalBases, {
                                stage: `${phaseLabel}Base ${idx + 1}/${totalBases} (NN Epoch ${ep}/${tot} - Loss: ${logs.loss.toFixed(4)})`,
                                loss: logs.loss
                            });
                        }
                    });
                } else if (mId === 'rf') {
                    const trees = capacity === 'fast' ? 15 : (capacity === 'deep' ? 50 : (this.userHyperparams.rfTrees || 30));
                    const depth = capacity === 'fast' ? 4 : (capacity === 'deep' ? 8 : (this.userHyperparams.rfDepth || 6));
                    subModel = new RandomForest(this.isClassification, trees, depth, 2, 'sqrt', subSeed);
                    await subModel.fitAsync(trainX, trainY, {
                        // Re-report progress at each yield so a cancelling onProgress can stop training here
                        onYield: (t) => onProgress && onProgress(idx + 1, totalBases, { stage: `${phaseLabel}Base ${idx + 1}/${totalBases} (RF tree ${t}/${trees})` })
                    });
                } else if (mId === 'glm') {
                    const epochs = capacity === 'fast' ? 80 : (capacity === 'deep' ? 250 : 150);
                    subModel = new GeneralizedLinearModel(this.isClassification, this.numClasses, 0.05, epochs, 0.01, 'momentum', subSeed);
                    await subModel.fitAsync(trainX, trainY, {
                        onYield: (ep) => onProgress && onProgress(idx + 1, totalBases, { stage: `${phaseLabel}Base ${idx + 1}/${totalBases} (GLM iteration ${ep}/${epochs})` })
                    });
                } else if (mId === 'knn') {
                    const kVal = capacity === 'fast' ? 3 : (capacity === 'deep' ? 7 : (this.userHyperparams.knnK || 5));
                    subModel = new KNNModel(this.isClassification, kVal, 'euclidean', 'distance', subSeed);
                    subModel.fit(trainX, trainY);
                } else if (mId === 'nb' && this.isClassification) {
                    subModel = new GaussianNaiveBayes(1e-4, 'empirical', subSeed);
                    subModel.fit(trainX, trainY);
                }

                if (subModel) {
                    if (subModel.usedFallback) {
                        this.fallbackNote = `Ensemble base neural network fell back to a linear model (${subModel.fallbackReason}).`;
                    }
                    models.push({ id: mId, model: subModel });
                }
                // Yield to the browser between base models so the UI (and Force Stop) stays responsive
                await new Promise(r => setTimeout(r, 0));
            }
            return models;
        }

        /**
         * testX / testY are accepted for API compatibility but are deliberately NOT used:
         * performance weights are estimated on an internal validation split of the TRAINING data,
         * so the test set never influences the model.
         */
        async fit(trainX, trainY, testX = null, testY = null, onProgress = null) {
            this.models = [];
            this.weights = [];
            this.baseMetrics = [];
            this.weightingNote = null;
            this.fallbackNote = null;
            this.rng = createPRNG(this.seed);

            const power = this.userHyperparams.power || 2;
            let rawWeights = null;
            const n = trainX.length;

            if (this.voting === 'weighted' && n >= 10) {
                // Internal 80/20 validation split of the training data (stratified by class when classifying)
                const idx = Array.from({ length: n }, (_, i) => i);
                for (let i = n - 1; i > 0; i--) {
                    const j = Math.floor(this.rng() * (i + 1));
                    [idx[i], idx[j]] = [idx[j], idx[i]];
                }
                const innerTrain = [];
                const innerVal = [];
                if (this.isClassification) {
                    const byClass = {};
                    idx.forEach(i => { (byClass[trainY[i]] = byClass[trainY[i]] || []).push(i); });
                    Object.values(byClass).forEach(list => {
                        const cut = list.length > 1 ? Math.max(1, Math.round(list.length * 0.8)) : list.length;
                        innerTrain.push(...list.slice(0, cut));
                        innerVal.push(...list.slice(cut));
                    });
                } else {
                    const cut = Math.round(n * 0.8);
                    innerTrain.push(...idx.slice(0, cut));
                    innerVal.push(...idx.slice(cut));
                }

                if (innerVal.length > 0 && innerTrain.length > 1) {
                    const itX = innerTrain.map(i => trainX[i]);
                    const itY = innerTrain.map(i => trainY[i]);
                    const ivX = innerVal.map(i => trainX[i]);
                    const ivY = innerVal.map(i => trainY[i]);
                    const weightingModels = await this._trainBaseModels(itX, itY, onProgress, 'Weighting phase · ');
                    rawWeights = [];
                    weightingModels.forEach(item => {
                        const preds = item.model.predict(ivX);
                        if (this.isClassification) {
                            // Balanced accuracy (mean per-class recall) so a model that ignores a
                            // clinically important minority class is not rewarded
                            const perClass = {};
                            for (let i = 0; i < ivY.length; i++) {
                                const c = ivY[i];
                                if (!perClass[c]) perClass[c] = { n: 0, hit: 0 };
                                perClass[c].n++;
                                if (preds[i] === c) perClass[c].hit++;
                            }
                            const recalls = Object.values(perClass).map(o => o.hit / o.n);
                            const balAcc = recalls.reduce((a, b) => a + b, 0) / (recalls.length || 1);
                            const w = Math.pow(Math.max(0.05, balAcc), power);
                            rawWeights.push(w);
                            this.baseMetrics.push({ id: item.id, balancedAccuracy: balAcc, weight: w });
                        } else {
                            let sumSq = 0;
                            for (let i = 0; i < ivY.length; i++) sumSq += Math.pow(ivY[i] - preds[i], 2);
                            const mse = sumSq / ivY.length;
                            const w = Math.pow(1.0 / (mse + 0.001), power / 2);
                            rawWeights.push(w);
                            this.baseMetrics.push({ id: item.id, mse, weight: w });
                        }
                    });
                    this.weightingNote = `Weights estimated on an internal ${innerVal.length}-case validation split of the training data.`;
                    // Weighting-phase models are no longer needed: free their GPU memory
                    weightingModels.forEach(item => {
                        if (item.model && typeof item.model.dispose === 'function') item.model.dispose();
                    });
                }
            }

            // Final base models are always fitted on the full training data
            this.models = await this._trainBaseModels(trainX, trainY, onProgress, rawWeights ? 'Final fit · ' : '');

            if (!rawWeights || rawWeights.length !== this.models.length) {
                if (this.voting === 'weighted') {
                    this.weightingNote = 'Too few training cases for an internal validation split; equal weights used.';
                }
                this.baseMetrics = [];
                rawWeights = new Array(this.models.length).fill(1.0);
                this.models.forEach(item => {
                    this.baseMetrics.push({ id: item.id, weight: 1.0 });
                });
            }

            const totalW = rawWeights.reduce((a, b) => a + b, 0) || 1.0;
            this.weights = rawWeights.map(w => w / totalW);
            this.baseMetrics.forEach((m, i) => {
                m.normalizedWeight = Number(this.weights[i].toFixed(3));
            });
        }

        predictSingle(x) {
            if (!this.models || this.models.length === 0) {
                throw new Error("Ensemble model is not trained yet.");
            }
            return this._combine(this.models.map(item => item.model.predictSingle(x)));
        }

        // Batched prediction: each base model predicts all rows at once (one GPU pass for the neural network)
        predictOutputs(X) {
            if (!this.models || this.models.length === 0) {
                throw new Error("Ensemble model is not trained yet.");
            }
            const perModel = this.models.map(item => batchOutputs(item.model, X));
            return X.map((_, r) => this._combine(perModel.map(outs => outs[r])));
        }

        async predictOutputsAsync(X, opts = {}) {
            if (!this.models || this.models.length === 0) {
                throw new Error("Ensemble model is not trained yet.");
            }
            const perModel = [];
            for (const item of this.models) perModel.push(await batchOutputsAsync(item.model, X, opts));
            return X.map((_, r) => this._combine(perModel.map(outs => outs[r])));
        }

        // Combines one row's base-model outputs (baseResults[i] belongs to this.models[i])
        _combine(baseResults) {
            if (this.isClassification) {
                if (this.voting === 'hard') {
                    const votes = new Array(this.numClasses).fill(0);
                    this.models.forEach((item, i) => {
                        const res = baseResults[i];
                        const predClass = res.pred;
                        if (predClass >= 0 && predClass < this.numClasses) {
                            votes[predClass] += this.weights[i];
                        }
                    });

                    let bestClass = 0;
                    let maxVotes = -1;
                    votes.forEach((v, c) => {
                        if (v > maxVotes) {
                            maxVotes = v;
                            bestClass = c;
                        }
                    });

                    const totalVotes = votes.reduce((a, b) => a + b, 0) || 1;
                    const probMap = {};
                    votes.forEach((v, c) => {
                        probMap[c] = Number((v / totalVotes).toFixed(4));
                    });

                    return {
                        pred: bestClass,
                        probabilities: probMap,
                        confidence: Number((maxVotes / totalVotes).toFixed(4)),
                        probabilityType: 'vote',
                        ensembleDetails: this.baseMetrics
                    };
                } else {
                    // Soft voting (weighted probability average)
                    const combinedProbs = new Array(this.numClasses).fill(0);

                    this.models.forEach((item, i) => {
                        const res = baseResults[i];
                        const w = this.weights[i];

                        if (res.probabilities) {
                            for (let c = 0; c < this.numClasses; c++) {
                                const p = res.probabilities[c] !== undefined ? res.probabilities[c] : (res.pred === c ? 1 : 0);
                                combinedProbs[c] += w * p;
                            }
                        } else {
                            combinedProbs[res.pred] += w;
                        }
                    });

                    let bestClass = 0;
                    let maxP = -1;
                    const probMap = {};

                    combinedProbs.forEach((p, c) => {
                        const clampedP = Number(Math.max(0, Math.min(1, p)).toFixed(4));
                        probMap[c] = clampedP;
                        if (clampedP > maxP) {
                            maxP = clampedP;
                            bestClass = c;
                        }
                    });

                    return {
                        pred: bestClass,
                        probabilities: probMap,
                        confidence: Number(maxP.toFixed(4)),
                        probabilityType: 'blend',
                        ensembleDetails: this.baseMetrics
                    };
                }
            } else {
                // Regression: weighted average of continuous outputs
                let weightedSum = 0;
                const subPredictions = [];

                this.models.forEach((item, i) => {
                    const res = baseResults[i];
                    weightedSum += this.weights[i] * res.pred;
                    subPredictions.push({ id: item.id, pred: res.pred, weight: this.weights[i] });
                });

                return {
                    pred: Number(weightedSum.toFixed(4)),
                    confidence: null,
                    subPredictions,
                    ensembleDetails: this.baseMetrics
                };
            }
        }

        predict(X) {
            return this.predictOutputs(X).map(o => o.pred);
        }

        // Frees GPU memory held by neural-network base models
        dispose() {
            (this.models || []).forEach(item => {
                if (item.model && typeof item.model.dispose === 'function') item.model.dispose();
            });
            this.models = [];
        }
    }

    // ==========================================
    // 7. MODEL MANAGER & EVALUATION METRICS
    // ==========================================

    // ==========================================
    // GRADIENT-BOOSTED TREES (histogram GBM)
    // ==========================================
    // Second-order gradient boosting of depth-limited regression trees on quantile-binned features (as in
    // LightGBM / scikit-learn HistGradientBoosting): logistic loss (binary), softmax (multiclass) or squared
    // error (regression); leaf values -G/(H+lambda); row subsampling; early stopping on an internal,
    // stratified validation slice of the training data (the test set is never used).
    class GradientBoostingModel {
        constructor(isClassification = true, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = isClassification ? Math.max(2, numClasses) : 1;
            this.nEstimators = config.nEstimators || 200;
            this.learningRate = config.learningRate || 0.1;
            this.maxDepth = config.maxDepth || 3;
            this.minSamplesLeaf = config.minSamplesLeaf || 5;
            this.lambda = config.l2 !== undefined ? config.l2 : 1;
            this.subsample = config.subsample || 0.8;
            this.maxBins = config.maxBins || 32;
            this.earlyStopping = config.earlyStopping !== false;
            this.validationFraction = config.validationFraction || 0.15;
            this.patience = config.patience || 15;
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.trees = [];       // rounds x K trees
            this.init = [];        // initial raw scores
            this.binEdges = [];    // per feature: thresholds between bins
            this.featureImportances = [];
            this.trainingInfo = null;
        }

        get K() { return this.isClassification && this.numClasses > 2 ? this.numClasses : 1; }

        _bin(X) {
            const p = X[0].length;
            this.binEdges = [];
            for (let j = 0; j < p; j++) {
                const vals = X.map(r => r[j]).sort((a, b) => a - b);
                const uniq = [...new Set(vals)];
                let edges;
                if (uniq.length <= this.maxBins) edges = uniq.slice(0, -1).map((v, i) => (v + uniq[i + 1]) / 2);
                else {
                    edges = [];
                    for (let b = 1; b < this.maxBins; b++) {
                        const q = vals[Math.floor((b * vals.length) / this.maxBins)];
                        const prev = vals[Math.floor((b * vals.length) / this.maxBins) - 1];
                        const e = (q + prev) / 2;
                        if (!edges.length || e > edges[edges.length - 1]) edges.push(e);
                    }
                }
                this.binEdges.push(edges);
            }
        }

        _binIndex(j, v) {
            const e = this.binEdges[j];
            let lo = 0, hi = e.length;
            while (lo < hi) { const mid = (lo + hi) >> 1; if (v <= e[mid]) hi = mid; else lo = mid + 1; }
            return lo;
        }

        _buildTree(B, rows, g, h, depth, gainAcc) {
            let G = 0, H = 0;
            rows.forEach(i => { G += g[i]; H += h[i]; });
            const leaf = { leaf: true, v: -G / (H + this.lambda) };
            if (depth >= this.maxDepth || rows.length < 2 * this.minSamplesLeaf) return leaf;
            const p = this.binEdges.length;
            const parentScore = (G * G) / (H + this.lambda);
            let best = null;
            for (let j = 0; j < p; j++) {
                const nb = this.binEdges[j].length + 1;
                if (nb < 2) continue;
                const hg = new Float64Array(nb), hh = new Float64Array(nb), hc = new Int32Array(nb);
                rows.forEach(i => { const b = B[i][j]; hg[b] += g[i]; hh[b] += h[i]; hc[b]++; });
                let gl = 0, hl = 0, cl = 0;
                for (let b = 0; b < nb - 1; b++) {
                    gl += hg[b]; hl += hh[b]; cl += hc[b];
                    const cr = rows.length - cl;
                    if (cl < this.minSamplesLeaf || cr < this.minSamplesLeaf) continue;
                    const gr = G - gl, hr = H - hl;
                    const gain = (gl * gl) / (hl + this.lambda) + (gr * gr) / (hr + this.lambda) - parentScore;
                    if (gain > 1e-12 && (!best || gain > best.gain)) best = { gain, j, b };
                }
            }
            if (!best) return leaf;
            gainAcc[best.j] += best.gain;
            const left = [], right = [];
            rows.forEach(i => (B[i][best.j] <= best.b ? left : right).push(i));
            return {
                leaf: false, f: best.j, t: this.binEdges[best.j][best.b],
                l: this._buildTree(B, left, g, h, depth + 1, gainAcc),
                r: this._buildTree(B, right, g, h, depth + 1, gainAcc)
            };
        }

        _treeValue(node, x) {
            while (!node.leaf) node = x[node.f] <= node.t ? node.l : node.r;
            return node.v;
        }

        _raw(x, nRounds = this.trees.length) {
            const f = this.init.slice();
            for (let m = 0; m < nRounds; m++) for (let k = 0; k < this.K; k++) f[k] += this.learningRate * this._treeValue(this.trees[m][k], x);
            return f;
        }

        _loss(F, y) {
            // mean loss of raw scores F (n x K) against labels y
            let L = 0;
            for (let i = 0; i < y.length; i++) {
                if (!this.isClassification) L += (F[i][0] - y[i]) ** 2;
                else if (this.K === 1) { const p = 1 / (1 + Math.exp(-F[i][0])); L -= y[i] === 1 ? Math.log(Math.max(p, 1e-15)) : Math.log(Math.max(1 - p, 1e-15)); }
                else { const mx = Math.max(...F[i]); const z = F[i].reduce((a, v) => a + Math.exp(v - mx), 0); L -= F[i][y[i]] - mx - Math.log(z); }
            }
            return L / y.length;
        }

        fit(X, y) { for (const _ of this._fitGen(X, y)) { /* run to completion */ } }

        async fitAsync(X, y, opts = {}) {
            let n = 0;
            for (const _ of this._fitGen(X, y)) {
                if (++n % 5 === 0) { await yieldToBrowser(); if (typeof opts.onYield === 'function') opts.onYield(n); }
            }
        }

        *_fitGen(X, y) {
            const rng = createPRNG(this.seed);
            const nAll = X.length;
            // Internal validation slice for early stopping (stratified for classification)
            let trI = Array.from({ length: nAll }, (_, i) => i), vaI = [];
            if (this.earlyStopping && nAll >= 40) {
                const groups = {};
                trI.forEach(i => { const key = this.isClassification ? y[i] : 0; (groups[key] = groups[key] || []).push(i); });
                trI = []; vaI = [];
                Object.values(groups).forEach(list => {
                    for (let a = list.length - 1; a > 0; a--) { const b = Math.floor(rng() * (a + 1)); [list[a], list[b]] = [list[b], list[a]]; }
                    const nv = Math.round(list.length * this.validationFraction);
                    vaI.push(...list.slice(0, nv)); trI.push(...list.slice(nv));
                });
            }
            const Xtr = trI.map(i => X[i]), ytr = trI.map(i => y[i]);
            this._bin(Xtr);
            const p = X[0].length;
            const Btr = Xtr.map(r => r.map((v, j) => this._binIndex(j, v)));
            const K = this.K, n = Xtr.length;
            // Initial scores: log-odds / log priors / mean
            if (!this.isClassification) this.init = [ytr.reduce((a, b) => a + b, 0) / n];
            else if (K === 1) { const pr = Math.min(1 - 1e-6, Math.max(1e-6, ytr.filter(v => v === 1).length / n)); this.init = [Math.log(pr / (1 - pr))]; }
            else this.init = Array.from({ length: K }, (_, k) => Math.log(Math.max(1e-6, ytr.filter(v => v === k).length / n)));
            const F = Xtr.map(() => this.init.slice());
            const Fva = vaI.map(() => this.init.slice());
            const gainAcc = new Float64Array(p);
            this.trees = [];
            let bestLoss = Infinity, bestRound = 0, sinceBest = 0;
            for (let m = 0; m < this.nEstimators; m++) {
                const round = [];
                // Gradients and hessians for every output
                const grads = [], hess = [];
                for (let k = 0; k < K; k++) { grads.push(new Float64Array(n)); hess.push(new Float64Array(n)); }
                for (let i = 0; i < n; i++) {
                    if (!this.isClassification) { grads[0][i] = F[i][0] - ytr[i]; hess[0][i] = 1; }
                    else if (K === 1) { const pr = 1 / (1 + Math.exp(-F[i][0])); grads[0][i] = pr - (ytr[i] === 1 ? 1 : 0); hess[0][i] = Math.max(pr * (1 - pr), 1e-12); }
                    else {
                        const mx = Math.max(...F[i]); const ex = F[i].map(v => Math.exp(v - mx)); const z = ex.reduce((a, b) => a + b, 0);
                        for (let k = 0; k < K; k++) { const pr = ex[k] / z; grads[k][i] = pr - (ytr[i] === k ? 1 : 0); hess[k][i] = Math.max(pr * (1 - pr), 1e-12); }
                    }
                }
                // Row subsample shared by the K trees of this round
                const rows = [];
                for (let i = 0; i < n; i++) if (this.subsample >= 1 || rng() < this.subsample) rows.push(i);
                for (let k = 0; k < K; k++) {
                    const tree = this._buildTree(Btr, rows.length ? rows : Array.from({ length: n }, (_, i) => i), grads[k], hess[k], 0, gainAcc);
                    round.push(tree);
                    for (let i = 0; i < n; i++) F[i][k] += this.learningRate * this._treeValue(tree, Xtr[i]);
                    for (let i = 0; i < vaI.length; i++) Fva[i][k] += this.learningRate * this._treeValue(tree, X[vaI[i]]);
                }
                this.trees.push(round);
                if (vaI.length) {
                    const L = this._loss(Fva, vaI.map(i => y[i]));
                    if (L < bestLoss - 1e-9) { bestLoss = L; bestRound = m + 1; sinceBest = 0; } else if (++sinceBest >= this.patience) break;
                } else bestRound = m + 1;
                yield m;
            }
            const roundsRun = this.trees.length;
            this.trees = this.trees.slice(0, Math.max(1, bestRound));
            const tot = gainAcc.reduce((a, b) => a + b, 0);
            this.featureImportances = Array.from(gainAcc, v => (tot > 0 ? Number((v / tot).toFixed(4)) : 0));
            this.trainingInfo = { fitCases: n, validationCases: vaI.length, roundsRun, bestRounds: this.trees.length, earlyStopped: roundsRun < this.nEstimators, maxRounds: this.nEstimators };
        }

        predictSingle(x) {
            const f = this._raw(x);
            if (!this.isClassification) return { pred: Number(f[0].toFixed(4)), confidence: null };
            let probs;
            if (this.K === 1) { const p1 = 1 / (1 + Math.exp(-f[0])); probs = { 0: 1 - p1, 1: p1 }; }
            else { const mx = Math.max(...f); const ex = f.map(v => Math.exp(v - mx)); const z = ex.reduce((a, b) => a + b, 0); probs = {}; ex.forEach((v, k) => { probs[k] = v / z; }); }
            let pred = 0;
            Object.keys(probs).forEach(k => { if (probs[k] > probs[pred]) pred = Number(k); });
            Object.keys(probs).forEach(k => { probs[k] = Number(probs[k].toFixed(4)); });
            return { pred, probabilities: probs, confidence: probs[pred], probabilityType: 'model' };
        }

        predict(X) { return X.map(x => this.predictSingle(x).pred); }
        predictOutputs(X) { return X.map(x => this.predictSingle(x)); }
    }


    // ==========================================
    // COX PROPORTIONAL-HAZARDS MODEL (time-to-event outcomes)
    // ==========================================
    // Partial likelihood with Efron's handling of tied event times, fitted by Newton-Raphson (optional small
    // ridge penalty for stability); hazard ratios with Wald 95% CIs; Breslow baseline cumulative hazard for
    // predicted survival S(t | x) = exp(-H0(t) * exp(x'b)).
    class CoxModel {
        constructor(config = {}) {
            this.penalty = config.penalty || 0;
            this.maxIter = config.maxIter || 50;
            this.beta = null;
            this.se = null;
            this.loglik = null;
            this.baseline = { times: [], cumHaz: [] };
            this.converged = false;
            this.note = null;
        }

        fit(X, time, event) {
            const n = X.length, p = X[0].length;
            const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => time[b] - time[a]); // descending time
            let beta = new Array(p).fill(0);
            let ll = -Infinity, info = null;
            const evalAt = (b) => {
                const eta = X.map(x => x.reduce((a, v, j) => a + v * b[j], 0));
                const w = eta.map(e => Math.exp(e));
                let loglik = 0;
                const grad = new Array(p).fill(0);
                const hess = Array.from({ length: p }, () => new Array(p).fill(0));
                let S0 = 0; const S1 = new Array(p).fill(0); const S2 = Array.from({ length: p }, () => new Array(p).fill(0));
                let k = 0;
                while (k < n) {
                    // all cases sharing this time enter the risk set together
                    const t = time[order[k]];
                    const group = [];
                    while (k < n && time[order[k]] === t) { group.push(order[k]); k++; }
                    group.forEach(i => {
                        S0 += w[i];
                        for (let a = 0; a < p; a++) { S1[a] += w[i] * X[i][a]; for (let c = a; c < p; c++) S2[a][c] += w[i] * X[i][a] * X[i][c]; }
                    });
                    const deaths = group.filter(i => event[i] === 1);
                    const d = deaths.length;
                    if (!d) continue;
                    let D0 = 0; const D1 = new Array(p).fill(0); const D2 = Array.from({ length: p }, () => new Array(p).fill(0));
                    deaths.forEach(i => {
                        D0 += w[i]; loglik += eta[i];
                        for (let a = 0; a < p; a++) { D1[a] += w[i] * X[i][a]; grad[a] += X[i][a]; for (let c = a; c < p; c++) D2[a][c] += w[i] * X[i][a] * X[i][c]; }
                    });
                    for (let l = 0; l < d; l++) {
                        const f = l / d;
                        const s0 = S0 - f * D0;
                        loglik -= Math.log(s0);
                        const s1 = S1.map((v, a) => v - f * D1[a]);
                        for (let a = 0; a < p; a++) {
                            grad[a] -= s1[a] / s0;
                            for (let c = a; c < p; c++) hess[a][c] -= (S2[a][c] - f * D2[a][c]) / s0 - (s1[a] * s1[c]) / (s0 * s0);
                        }
                    }
                }
                for (let a = 0; a < p; a++) for (let c = 0; c < a; c++) hess[a][c] = hess[c][a];
                if (this.penalty > 0) for (let a = 0; a < p; a++) { loglik -= 0.5 * this.penalty * b[a] * b[a]; grad[a] -= this.penalty * b[a]; hess[a][a] -= this.penalty; }
                return { loglik, grad, hess };
            };
            for (let it = 0; it < this.maxIter; it++) {
                const r = evalAt(beta);
                const negH = r.hess.map(row => row.map(v => -v));
                const step = solveLinear(negH, r.grad);
                if (!step) { this.note = 'The predictors are collinear, so the Cox model cannot be estimated; try fewer predictors.'; break; }
                // step halving keeps the partial likelihood increasing
                let t = 1, nb = beta.map((b, j) => b + step[j]), r2 = evalAt(nb);
                while (r2.loglik < r.loglik - 1e-10 && t > 1e-4) { t /= 2; nb = beta.map((b, j) => b + t * step[j]); r2 = evalAt(nb); }
                beta = nb; ll = r2.loglik; info = r2.hess.map(row => row.map(v => -v));
                if (beta.some(b => !isFinite(b) || Math.abs(b) > 30)) { this.note = 'The model did not converge: a predictor separates the event times (almost) perfectly.'; break; }
                if (Math.max(...step.map(v => Math.abs(v * t))) < 1e-9) { this.converged = true; break; }
            }
            this.beta = beta;
            this.loglik = ll;
            const inv = info ? invertMatrix(info) : null;
            this.se = inv ? inv.map((row, j) => Math.sqrt(Math.max(row[j], 0))) : beta.map(() => NaN);
            if (!this.converged && !this.note) this.note = 'The model did not converge within the iteration limit.';
            // Breslow baseline cumulative hazard at each distinct event time
            const risk = X.map(x => Math.exp(x.reduce((a, v, j) => a + v * beta[j], 0)));
            const eventTimes = [...new Set(time.filter((t, i) => event[i] === 1))].sort((a, b) => a - b);
            let H = 0;
            this.baseline = { times: [], cumHaz: [] };
            eventTimes.forEach(t => {
                let d = 0, s = 0;
                for (let i = 0; i < n; i++) { if (time[i] >= t) s += risk[i]; if (time[i] === t && event[i] === 1) d++; }
                H += d / s;
                this.baseline.times.push(t); this.baseline.cumHaz.push(H);
            });
            return this;
        }

        linearPredictor(x) { return x.reduce((a, v, j) => a + v * this.beta[j], 0); }

        cumHazAt(t) {
            const ts = this.baseline.times;
            let lo = 0, hi = ts.length;
            while (lo < hi) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid + 1; else hi = mid; }
            return lo === 0 ? 0 : this.baseline.cumHaz[lo - 1];
        }

        survivalAt(x, t) { return Math.exp(-this.cumHazAt(t) * Math.exp(this.linearPredictor(x))); }

        summary(names) {
            return this.beta.map((b, j) => {
                const se = this.se[j], z = se > 0 ? b / se : null;
                return { name: names ? names[j] : `x${j + 1}`, coef: b, se, z, p: z === null ? null : 2 * (1 - normalCdf(Math.abs(z))), hr: Math.exp(b), hrLow: Math.exp(b - 1.959964 * se), hrHigh: Math.exp(b + 1.959964 * se) };
            });
        }
    }

    // ==========================================
    // RECALIBRATION WRAPPER (binary outcomes)
    // ==========================================
    // Applies a calibrator, fitted on out-of-fold training predictions, to the positive-class probability of a
    // trained model: Platt scaling (logistic regression on logit p) or isotonic regression (monotone step function).
    function logitClip(p) { const q = Math.min(1 - 1e-6, Math.max(1e-6, Number(p) || 0)); return Math.log(q / (1 - q)); }

    function applyCalibrator(cal, p) {
        if (!cal) return p;
        if (cal.type === 'platt') return 1 / (1 + Math.exp(-(cal.a + cal.b * logitClip(p))));
        if (cal.type === 'isotonic') {
            const xs = cal.x, ys = cal.y;
            if (!xs.length) return p;
            if (p <= xs[0]) return ys[0];
            if (p >= xs[xs.length - 1]) return ys[ys.length - 1];
            let lo = 0, hi = xs.length - 1;
            while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= p) lo = mid; else hi = mid; }
            const t = xs[hi] === xs[lo] ? 0 : (p - xs[lo]) / (xs[hi] - xs[lo]);
            return ys[lo] + t * (ys[hi] - ys[lo]);
        }
        return p;
    }

    class CalibratedModel {
        constructor(base, calibrator) {
            this.base = base;
            this.calibrator = calibrator;
            this.isClassification = true;
            this.seed = base ? base.seed : null;
            ['trainingInfo', 'usedFallback', 'fallbackReason', 'weightingNote', 'fallbackNote', 'featureImportances'].forEach(k => { if (base && base[k] !== undefined) this[k] = base[k]; });
        }
        _adjust(o) {
            if (!o || !o.probabilities) return o;
            const p1 = applyCalibrator(this.calibrator, Number(o.probabilities[1] || 0));
            const probabilities = { 0: 1 - p1, 1: p1 };
            const pred = p1 >= 0.5 ? 1 : 0;
            return { ...o, pred, probabilities, confidence: probabilities[pred], probabilityType: 'recalibrated' };
        }
        predictSingle(x) { return this._adjust(this.base.predictSingle(x)); }
        predict(X) { return X.map(x => this.predictSingle(x).pred); }
        predictOutputs(X) { return (typeof this.base.predictOutputs === 'function' ? this.base.predictOutputs(X) : X.map(x => this.base.predictSingle(x))).map(o => this._adjust(o)); }
        async predictOutputsAsync(X, opts) {
            const outs = typeof this.base.predictOutputsAsync === 'function' ? await this.base.predictOutputsAsync(X, opts) : this.predictOutputs(X);
            return typeof this.base.predictOutputsAsync === 'function' ? outs.map(o => this._adjust(o)) : outs;
        }
        dispose() { if (this.base && this.base.dispose) this.base.dispose(); }
    }


    // ==========================================
    // SHAPLEY-VALUE EXPLANATIONS (interventional, against a background of training patients)
    // ==========================================
    // The players are the original predictors (`groups[j]` = predictor of encoded column j), so a
    // categorical predictor's one-hot columns move together. v(S) = mean over background patients z of
    // f(x on the predictors in S, z elsewhere). The contributions add up exactly to
    // f(x) − mean f(z): the prediction minus the average prediction for the background patients.

    // Uniform view of one tree: RF nodes {isLeaf, value, featureIdx, threshold} or boosted {leaf, v, f, t}
    function treeView(node, leafValue) {
        if (node.isLeaf !== undefined) return node.isLeaf ? { leaf: true, v: leafValue(node.value) } : { leaf: false, f: node.featureIdx, t: node.threshold, l: treeView(node.left, leafValue), r: treeView(node.right, leafValue) };
        return node.leaf ? { leaf: true, v: leafValue(node.v) } : { leaf: false, f: node.f, t: node.t, l: treeView(node.l, leafValue), r: treeView(node.r, leafValue) };
    }

    const factorial = (() => { const f = [1]; for (let i = 1; i <= 170; i++) f[i] = f[i - 1] * i; return n => f[n]; })();

    // Exact Shapley values of one tree for patient x against one background patient z. A leaf is reached by
    // the hybrid patient exactly when the predictors on which the path follows x (where x and z differ) are
    // in S and those on which it follows z are not; that game has a closed-form Shapley value.
    function treeShapPair(tree, x, z, groups, phi) {
        const walk = (node, A, B) => {
            if (node.leaf) {
                const a = A.length, b = B.length;
                if (a + b === 0 || node.v === 0) return;
                const d = factorial(a + b);
                if (a) { const w = node.v * factorial(a - 1) * factorial(b) / d; A.forEach(g => { phi[g] += w; }); }
                if (b) { const w = node.v * factorial(a) * factorial(b - 1) / d; B.forEach(g => { phi[g] -= w; }); }
                return;
            }
            const g = groups[node.f];
            const nx = x[node.f] <= node.t ? node.l : node.r;
            const nz = z[node.f] <= node.t ? node.l : node.r;
            if (nx === nz) return walk(nx, A, B);
            if (A.includes(g)) return walk(nx, A, B);
            if (B.includes(g)) return walk(nz, A, B);
            walk(nx, [...A, g], B);
            walk(nz, A, [...B, g]);
        };
        walk(tree, [], []);
    }

    // Tree models: the trees of the output being explained, with their leaf values on that output's scale
    function explainableTrees(model, classIdx) {
        const m = model && model.base && model.calibrator ? model.base : model; // a recalibrated model: explain its base
        if (m instanceof RandomForest) {
            const T = m.trees.length;
            const leaf = m.isClassification ? (v => (Number(v) === classIdx ? 1 / T : 0)) : (v => v / T);
            return { trees: m.trees.map(t => treeView(t.root, leaf)), scale: m.isClassification ? 'probability' : 'outcome', offset: 0, recalibrated: m !== model };
        }
        if (m instanceof GradientBoostingModel) {
            const k = m.K === 1 ? 0 : classIdx;
            return {
                trees: m.trees.map(round => treeView(round[k], v => m.learningRate * v)),
                scale: m.isClassification ? 'log-odds' : 'outcome', offset: m.init[k], recalibrated: m !== model, multiclass: m.K > 1
            };
        }
        return null;
    }

    // Exact interventional TreeSHAP; returns null when the model is not a tree model
    function treeShap(model, x, background, groups, nGroups, classIdx) {
        const tv = explainableTrees(model, classIdx);
        if (!tv || !background.length) return null;
        const phi = new Array(nGroups).fill(0);
        const tmp = new Array(nGroups).fill(0);
        let baseSum = 0;
        const value = tree => z => { let n = tree; while (!n.leaf) n = z[n.f] <= n.t ? n.l : n.r; return n.v; };
        background.forEach(z => {
            tmp.fill(0);
            tv.trees.forEach(tree => { treeShapPair(tree, x, z, groups, tmp); baseSum += value(tree)(z); });
            for (let g = 0; g < nGroups; g++) phi[g] += tmp[g] / background.length;
        });
        const fx = tv.offset + tv.trees.reduce((a, tree) => a + value(tree)(x), 0);
        return { phi, base: tv.offset + baseSum / background.length, value: fx, scale: tv.scale, exact: true, recalibrated: tv.recalibrated, multiclass: Boolean(tv.multiclass) };
    }

    // Any model: Shapley values estimated by sampling orderings of the predictors (Strumbelj & Kononenko),
    // each with a random background patient; antithetic pairs (an ordering and its reverse) reduce noise.
    // output(rows) returns the explained output for a batch of encoded rows.
    function sampledShapley(output, x, background, groups, nGroups, nOrderings = 200, seed = 1) {
        const rng = createPRNG(seed);
        const cols = Array.from({ length: nGroups }, () => []);
        groups.forEach((g, j) => cols[g].push(j));
        const rows = [], plan = [];
        for (let m = 0; m < nOrderings; m += 2) {
            const order = Array.from({ length: nGroups }, (_, i) => i);
            for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
            const z = background[Math.floor(rng() * background.length)];
            [order, order.slice().reverse()].forEach(ord => {
                const h = z.slice();
                const start = rows.length;
                rows.push(h.slice());
                ord.forEach(g => { cols[g].forEach(j => { h[j] = x[j]; }); rows.push(h.slice()); });
                plan.push({ ord, start });
            });
        }
        const out = output(rows);
        const phi = new Array(nGroups).fill(0);
        let base = 0;
        plan.forEach(({ ord, start }) => {
            base += out[start];
            ord.forEach((g, i) => { phi[g] += out[start + i + 1] - out[start + i]; });
        });
        const n = plan.length;
        return { phi: phi.map(v => v / n), base: base / n, value: out[plan[0].start + nGroups], exact: false, orderings: n };
    }

    const ModelEngine = {
        treeShap,
        sampledShapley,
        SUPPORTED_MODELS: [
            { id: 'ensemble', name: 'Voting / Performance-Weighted Ensemble', type: 'both', desc: 'Combines the selected base models (neural network, random forest, logistic/ridge, k-NN, naive Bayes) by soft probability averaging, hard majority voting, or weights learned on an internal validation split of the training data. Not a cross-validated stacked Super Learner.' },
            { id: 'nn', name: 'Neural Network (Multilayer Perceptron, TF.js)', type: 'both', desc: 'Multilayer perceptron with dense hidden layers, dropout, L2 weight decay and early stopping on an internal validation split. Falls back to a linear model (and says so) if TensorFlow.js fails.' },
            { id: 'rf', name: 'Random Forest (Ensemble Trees)', type: 'both', desc: 'Bootstrap-sampled decision trees with random feature subsets. Splits use Gini impurity (classification) or variance reduction (regression), scanning every threshold for cohorts under 5,000 patients (an approximate grid of about 10 thresholds per feature for larger ones). Feature importance is impurity-based (MDI): useful for ranking, not evidence of causal or clinical effect.' },
            { id: 'glm', name: 'Logistic / Ridge Regression', type: 'both', desc: 'L2-regularized linear (ridge) or logistic/softmax model fitted by gradient descent on standardised features.' },
            { id: 'knn', name: 'k-Nearest Neighbors (k-NN)', type: 'both', desc: 'Non-parametric distance-based model; "probabilities" are (weighted) neighbour vote shares.' },
            { id: 'nb', name: 'Naive Bayes (Gaussian + Bernoulli)', type: 'classification', desc: 'Naive Bayes with Gaussian likelihoods for continuous features and Laplace-smoothed Bernoulli likelihoods for 0/1 indicator features (one-hot categories, binary flags).' }
        ],

        /**
         * Create an instance of a model based on user selection
         */
        createModel(modelId, isClassification, numClasses = 2, userHyperparams = {}) {
            let model;
            const seed = userHyperparams.seed !== undefined && userHyperparams.seed !== null ? userHyperparams.seed : null;
            if (modelId === 'ensemble') {
                model = new EnsembleModel(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'nn') {
                model = new DeepNeuralNetwork(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'rf') {
                const trees = userHyperparams.numTrees || 30;
                const depth = userHyperparams.maxDepth || 6;
                const minSplit = userHyperparams.minSamplesSplit || 2;
                const featRatio = userHyperparams.featureRatio || 'sqrt';
                model = new RandomForest(isClassification, trees, depth, minSplit, featRatio, seed, userHyperparams.splitMode || 'auto');
            } else if (modelId === 'glm') {
                model = new GeneralizedLinearModel(
                    isClassification,
                    numClasses,
                    userHyperparams.lr || 0.05,
                    userHyperparams.epochs || 150,
                    userHyperparams.lambdaL2 !== undefined ? userHyperparams.lambdaL2 : 0.01,
                    userHyperparams.solver || 'momentum',
                    seed
                );
            } else if (modelId === 'knn') {
                model = new KNNModel(
                    isClassification,
                    userHyperparams.k || 5,
                    userHyperparams.metric || 'euclidean',
                    userHyperparams.weighting || 'uniform',
                    seed
                );
            } else if (modelId === 'gbm') {
                model = new GradientBoostingModel(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'nb') {
                if (!isClassification) throw new Error("Naive Bayes is only supported for classification tasks.");
                model = new GaussianNaiveBayes(
                    userHyperparams.varSmoothing !== undefined ? userHyperparams.varSmoothing : 1e-4,
                    userHyperparams.priorMode || 'empirical',
                    seed
                );
            } else {
                throw new Error(`Unknown model ID: ${modelId}`);
            }
            if (model && seed !== null && seed !== undefined) {
                model.seed = seed;
            }
            return model;
        },

        /**
         * Evaluate model performance on a held-out set.
         * options.regressionThreshold: cutoff for the exploratory 2×2 table of a continuous outcome.
         *   It should come from the TRAINING data (or be a prespecified clinical cutoff); if omitted,
         *   no 2×2 table is produced.
         * options.thresholdSource: free-text description of where that cutoff came from.
         */
        evaluate(model, testX, testY, isClassification, targetClasses = [], options = {}) {
            if (!testX || testX.length === 0) {
                return null;
            }

            if (isClassification) {
                const numClasses = targetClasses.length || [...new Set(testY)].length;
                const total = testX.length;
                // options.outputs: predictions already computed (e.g. asynchronously by evaluateAsync)
                let outputs = options.outputs || batchOutputs(model, testX);
                // Optional decision threshold for binary outcomes (default: the model's own 0.5 / arg-max rule)
                const dt = Number(options.decisionThreshold);
                if (numClasses === 2 && isFinite(dt) && dt > 0 && dt < 1 && Math.abs(dt - 0.5) > 1e-9) {
                    outputs = outputs.map(o => ModelEngine.applyDecisionThreshold(o, dt));
                }
                const preds = outputs.map(o => o.pred);
                let correct = 0;

                // Confusion Matrix [actual][pred]
                const confMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));

                for (let i = 0; i < total; i++) {
                    const actual = testY[i];
                    const pred = preds[i];
                    if (actual === pred) correct++;
                    if (actual < numClasses && pred < numClasses) {
                        confMatrix[actual][pred]++;
                    }
                }

                const accuracyRaw = correct / total;
                const accuracy = Number(accuracyRaw.toFixed(4));

                // Wilson score 95% confidence interval for accuracy
                const z = 1.959964;
                const denom = 1 + (z * z) / total;
                const centre = (accuracyRaw + (z * z) / (2 * total)) / denom;
                const half = (z * Math.sqrt((accuracyRaw * (1 - accuracyRaw)) / total + (z * z) / (4 * total * total))) / denom;
                const accuracyCI = [Number(Math.max(0, centre - half).toFixed(4)), Number(Math.min(1, centre + half).toFixed(4))];

                // Per-class metrics
                const classList = targetClasses.length ? targetClasses : Array.from({ length: numClasses }, (_, i) => String(i));
                const classMetrics = classList.map((clsName, cIdx) => {
                    let tp = confMatrix[cIdx][cIdx];
                    let fp = 0;
                    let fn = 0;
                    let support = 0;

                    for (let r = 0; r < numClasses; r++) {
                        support += confMatrix[cIdx][r];
                        if (r !== cIdx) {
                            fn += confMatrix[cIdx][r];
                            fp += confMatrix[r][cIdx];
                        }
                    }

                    const precision = (tp + fp) > 0 ? Number((tp / (tp + fp)).toFixed(4)) : 0;
                    const recall = (tp + fn) > 0 ? Number((tp / (tp + fn)).toFixed(4)) : 0;
                    const f1 = (precision + recall) > 0 ? Number(((2 * precision * recall) / (precision + recall)).toFixed(4)) : 0;

                    return {
                        className: clsName,
                        precision,
                        recall,
                        f1,
                        support
                    };
                });

                // Macro-averaged metrics (all classes, as in scikit-learn's default)
                const macroPrecision = Number((classMetrics.reduce((a, b) => a + b.precision, 0) / numClasses).toFixed(4));
                const macroRecall = Number((classMetrics.reduce((a, b) => a + b.recall, 0) / numClasses).toFixed(4));
                const macroF1 = Number((classMetrics.reduce((a, b) => a + b.f1, 0) / numClasses).toFixed(4));
                // Balanced accuracy: mean recall over classes present in the test set
                const present = classMetrics.filter(c => c.support > 0);
                const balancedAccuracy = present.length ? Number((present.reduce((a, b) => a + b.recall, 0) / present.length).toFixed(4)) : null;
                const classesMissingFromTest = classMetrics.filter(c => c.support === 0).map(c => c.className);

                // Class-probability vectors (vote shares for RF / hard voting; neighbour weights for k-NN)
                const probVectors = outputs.map(o => {
                    const v = new Array(numClasses).fill(0);
                    if (o.probabilities) {
                        for (let c = 0; c < numClasses; c++) {
                            const p = Number(o.probabilities[c]);
                            v[c] = isNaN(p) ? 0 : p;
                        }
                    } else if (o.pred >= 0 && o.pred < numClasses) {
                        v[o.pred] = 1;
                    }
                    return v;
                });

                // Brier score (lower is better): binary = mean (p_positive − y)²; multiclass = mean Σ_c (p_c − y_c)²
                let brierScore = null;
                if (numClasses === 2) {
                    brierScore = probVectors.reduce((acc, v, i) => acc + Math.pow(v[1] - (testY[i] === 1 ? 1 : 0), 2), 0) / total;
                } else if (numClasses > 2) {
                    brierScore = probVectors.reduce((acc, v, i) => acc + v.reduce((s2, p, c) => s2 + Math.pow(p - (testY[i] === c ? 1 : 0), 2), 0), 0) / total;
                }
                if (brierScore !== null) brierScore = Number(brierScore.toFixed(4));

                // ROC-AUC for binary classification
                let rocAuc = null;
                if (numClasses === 2) {
                    try {
                        rocAuc = this._calculateBinaryAUC(testY, probVectors.map(v => v[1]));
                    } catch (e) {
                        rocAuc = null;
                    }
                }

                const probabilityType = outputs.length ? (outputs[0].probabilityType || 'model') : 'model';
                const probPositive = numClasses === 2 ? probVectors.map(v => Number(v[1].toFixed(4))) : null;
                const probMatrix = numClasses > 2 ? probVectors.map(v => v.map(q => Number(q.toFixed(4)))) : null;
                const calibration = numClasses === 2 ? this.calibrationSummary(testY.map(v => (v === 1 ? 1 : 0)), probVectors.map(v => v[1])) : null;

                return {
                    isClassification: true,
                    totalSamples: total,
                    accuracy,
                    accuracyCI,
                    balancedAccuracy,
                    macroPrecision,
                    macroRecall,
                    macroF1,
                    rocAuc,
                    brierScore,
                    probabilityType,
                    calibration,
                    probPositive,
                    probMatrix,
                    numClasses,
                    decisionThreshold: numClasses === 2 && isFinite(dt) && dt > 0 && dt < 1 ? dt : (numClasses === 2 ? 0.5 : null),
                    classesMissingFromTest,
                    // For binary outcomes, class index 1 is the "positive" class for AUC / sensitivity / TP
                    positiveClass: numClasses === 2 && targetClasses.length === 2 ? targetClasses[1] : null,
                    confMatrix,
                    classMetrics,
                    actuals: testY,
                    predictions: preds,
                    seed: model.seed !== undefined ? model.seed : null
                };
            } else {
                // Regression Metrics: R2, MSE, RMSE, MAE
                const preds = options.outputs ? options.outputs.map(o => o.pred) : model.predict(testX);
                const total = testX.length;
                let sumSqErr = 0;
                let sumAbsErr = 0;
                const meanActual = testY.reduce((a, b) => a + b, 0) / total;
                let totalSq = 0;

                for (let i = 0; i < total; i++) {
                    const actual = testY[i];
                    const pred = preds[i];
                    const err = actual - pred;
                    sumSqErr += err * err;
                    sumAbsErr += Math.abs(err);
                    totalSq += Math.pow(actual - meanActual, 2);
                }

                const mse = Number((sumSqErr / total).toFixed(4));
                const rmse = Number((Math.sqrt(sumSqErr / total)).toFixed(4));
                const mae = Number((sumAbsErr / total).toFixed(4));
                // R² is undefined when the test outcomes have no variance
                const r2 = totalSq > 0 ? Number((1 - (sumSqErr / totalSq)).toFixed(4)) : null;

                const result = {
                    isClassification: false,
                    totalSamples: total,
                    r2,
                    mse,
                    rmse,
                    mae,
                    actuals: testY,
                    predictions: preds,
                    seed: model.seed !== undefined ? model.seed : null
                };

                // Exploratory 2×2 dichotomisation at a cutoff fixed BEFORE looking at the test outcomes
                const threshold = options.regressionThreshold;
                if (typeof threshold === 'number' && isFinite(threshold)) {
                    // Row 0 = Actual < cutoff, Row 1 = Actual >= cutoff; Col 0/1 likewise for predictions
                    const confMatrix2x2 = [[0, 0], [0, 0]];
                    let hits = 0;
                    for (let i = 0; i < total; i++) {
                        const actBin = testY[i] >= threshold ? 1 : 0;
                        const predBin = preds[i] >= threshold ? 1 : 0;
                        confMatrix2x2[actBin][predBin]++;
                        if (actBin === predBin) hits++;
                    }
                    const tp = confMatrix2x2[1][1];
                    const fn = confMatrix2x2[1][0];
                    const fp = confMatrix2x2[0][1];
                    const tn = confMatrix2x2[0][0];
                    result.confMatrix = confMatrix2x2;
                    result.threshold = Number(threshold.toFixed(4));
                    result.thresholdSource = options.thresholdSource || 'user-supplied cutoff';
                    result.thresholdAccuracy = Number((hits / total).toFixed(4));
                    // Undefined (null) when no test cases fall on that side of the cutoff
                    result.sensitivity = (tp + fn) > 0 ? Number((tp / (tp + fn)).toFixed(4)) : null;
                    result.specificity = (tn + fp) > 0 ? Number((tn / (tn + fp)).toFixed(4)) : null;
                }
                return result;
            }
        },

        predictOutputs(model, X) {
            return batchOutputs(model, X);
        },

        predictOutputsAsync(model, X, opts = {}) {
            return batchOutputsAsync(model, X, opts);
        },

        // evaluate() with predictions computed asynchronously (keeps the page responsive and lets
        // opts.onYield cancel); metrics are identical to evaluate()
        async evaluateAsync(model, testX, testY, isClassification, targetClasses = [], options = {}, opts = {}) {
            if (!testX || testX.length === 0) return null;
            const outputs = await batchOutputsAsync(model, testX, opts);
            return this.evaluate(model, testX, testY, isClassification, targetClasses, { ...options, outputs });
        },

        evaluateModel(model, testX, testY, isClassificationOrClasses = true, targetClassesOrStats = []) {
            let isClassification = true;
            let targetClasses = [];
            if (typeof isClassificationOrClasses === 'boolean') {
                isClassification = isClassificationOrClasses;
                targetClasses = Array.isArray(targetClassesOrStats) ? targetClassesOrStats : [];
            } else if (Array.isArray(isClassificationOrClasses)) {
                targetClasses = isClassificationOrClasses;
                isClassification = targetClasses.length > 0;
            } else if (model && model.isClassification !== undefined) {
                isClassification = Boolean(model.isClassification);
                targetClasses = Array.isArray(targetClassesOrStats) ? targetClassesOrStats : [];
            }
            return this.evaluate(model, testX, testY, isClassification, targetClasses);
        },

        // ---------------------------------------------------------------------------------------------
        // Uncertainty, model comparison and clinical-utility statistics (computed on held-out predictions)
        // ---------------------------------------------------------------------------------------------

        /** ROC curve points (FPR, TPR) from binary labels (1 = positive) and scores; ties handled as one step */
        rocCurve(y, p) {
            const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
            const P = y.reduce((a, b) => a + b, 0), N = y.length - P;
            const pts = [{ fpr: 0, tpr: 0 }];
            let tp = 0, fp = 0;
            for (let k = 0; k < order.length; k++) {
                const i = order[k];
                if (y[i] === 1) tp++; else fp++;
                if (k === order.length - 1 || p[order[k + 1]] !== p[i]) pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0 });
            }
            return pts;
        },

        /** Precision-recall curve points and average precision (step-wise, as in scikit-learn) */
        prCurve(y, p) {
            const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
            const P = y.reduce((a, b) => a + b, 0);
            const pts = [];
            let tp = 0, fp = 0, ap = 0, prevRecall = 0;
            for (let k = 0; k < order.length; k++) {
                const i = order[k];
                if (y[i] === 1) tp++; else fp++;
                if (k === order.length - 1 || p[order[k + 1]] !== p[i]) {
                    const recall = P ? tp / P : 0, precision = tp / (tp + fp);
                    ap += (recall - prevRecall) * precision;
                    prevRecall = recall;
                    pts.push({ recall, precision });
                }
            }
            return { points: pts, averagePrecision: P ? ap : null, prevalence: y.length ? P / y.length : null };
        },

        /**
         * DeLong test for the difference between two correlated ROC AUCs measured on the same patients
         * (DeLong, DeLong & Clarke-Pearson 1988). Returns both AUCs, the difference, its SE, z and two-sided p.
         */
        delongTest(y, pA, pB) {
            const pos = [], neg = [];
            y.forEach((v, i) => (v === 1 ? pos : neg).push(i));
            const m = pos.length, n = neg.length;
            if (m < 2 || n < 2) return null;
            const psi = (a, b) => (a > b ? 1 : a === b ? 0.5 : 0);
            const comps = [pA, pB].map(sc => {
                const V10 = pos.map(i => neg.reduce((s, j) => s + psi(sc[i], sc[j]), 0) / n);
                const V01 = neg.map(j => pos.reduce((s, i) => s + psi(sc[i], sc[j]), 0) / m);
                const auc = V10.reduce((a, b) => a + b, 0) / m;
                return { V10, V01, auc };
            });
            const cov = (a, b, len) => {
                const ma = a.reduce((s, v) => s + v, 0) / len, mb = b.reduce((s, v) => s + v, 0) / len;
                return a.reduce((s, v, k) => s + (v - ma) * (b[k] - mb), 0) / (len - 1);
            };
            const [A, B] = comps;
            const varDiff = (cov(A.V10, A.V10, m) + cov(B.V10, B.V10, m) - 2 * cov(A.V10, B.V10, m)) / m
                + (cov(A.V01, A.V01, n) + cov(B.V01, B.V01, n) - 2 * cov(A.V01, B.V01, n)) / n;
            const diff = A.auc - B.auc;
            if (!(varDiff > 0)) return { aucA: A.auc, aucB: B.auc, diff, se: 0, z: null, p: diff === 0 ? 1 : null };
            const se = Math.sqrt(varDiff), z = diff / se;
            return { aucA: A.auc, aucB: B.auc, diff, se, z, p: 2 * (1 - normalCdf(Math.abs(z))) };
        },

        /** Decision curve analysis: net benefit of the model, "treat all" and "treat none" across thresholds */
        decisionCurve(y, p, thresholds = null) {
            const n = y.length;
            const prev = n ? y.reduce((a, b) => a + b, 0) / n : 0;
            const ths = thresholds || Array.from({ length: 19 }, (_, i) => Number(((i + 1) * 0.05).toFixed(2)));
            return ths.map(t => {
                let tp = 0, fp = 0;
                p.forEach((v, i) => { if (v >= t) { if (y[i] === 1) tp++; else fp++; } });
                const w = t / (1 - t);
                return { threshold: t, model: n ? tp / n - (fp / n) * w : 0, treatAll: prev - (1 - prev) * w, treatNone: 0 };
            });
        },

        /**
         * Percentile bootstrap 95% confidence intervals for the held-out metrics in an evaluation result,
         * resampling test patients with replacement (seeded, reproducible).
         */
        bootstrapCIs(ev, B = 1000, seed = 2024) {
            if (!ev || !Array.isArray(ev.actuals) || ev.actuals.length < 5) return null;
            const rng = createPRNG(seed);
            const n = ev.actuals.length;
            const y = ev.actuals, yhat = ev.predictions;
            const stats = {};
            const add = (k, v) => { if (typeof v === 'number' && isFinite(v)) (stats[k] = stats[k] || []).push(v); };
            const slopeB = Math.min(B, n > 3000 ? 0 : 300); // calibration slope needs an iterative fit: fewer resamples
            for (let b = 0; b < B; b++) {
                const idx = Array.from({ length: n }, () => Math.floor(rng() * n));
                if (ev.isClassification) {
                    const yy = idx.map(i => y[i]), pp = idx.map(i => yhat[i]);
                    const k = ev.numClasses || Math.max(...y) + 1;
                    let correct = 0; const tp = new Array(k).fill(0), sup = new Array(k).fill(0), prd = new Array(k).fill(0);
                    yy.forEach((v, i) => { sup[v]++; prd[pp[i]]++; if (v === pp[i]) { correct++; tp[v]++; } });
                    add('accuracy', correct / n);
                    const present = sup.map((s, c) => c).filter(c => sup[c] > 0);
                    add('balancedAccuracy', present.reduce((a, c) => a + tp[c] / sup[c], 0) / present.length);
                    add('macroF1', sup.reduce((a, s, c) => { const pr = prd[c] ? tp[c] / prd[c] : 0, rc = s ? tp[c] / s : 0; return a + (pr + rc ? 2 * pr * rc / (pr + rc) : 0); }, 0) / k);
                    if (ev.probPositive) {
                        const yb = yy.map(v => (v === 1 ? 1 : 0)), pr = idx.map(i => ev.probPositive[i]);
                        add('brierScore', pr.reduce((a, q, i) => a + (q - yb[i]) ** 2, 0) / n);
                        if (yb.some(v => v === 1) && yb.some(v => v === 0)) add('rocAuc', this._calculateBinaryAUC(yb, pr));
                        if (b < slopeB) { const c = this.calibrationSummary(yb, pr); add('calibrationSlope', c.slope); add('calibrationIntercept', c.intercept); }
                    } else if (ev.probMatrix) {
                        add('brierScore', idx.reduce((a, i) => a + ev.probMatrix[i].reduce((s, q, c) => s + (q - (y[i] === c ? 1 : 0)) ** 2, 0), 0) / n);
                    }
                } else {
                    const yy = idx.map(i => y[i]), pp = idx.map(i => yhat[i]);
                    const mean = yy.reduce((a, v) => a + v, 0) / n;
                    const sse = yy.reduce((a, v, i) => a + (v - pp[i]) ** 2, 0), sst = yy.reduce((a, v) => a + (v - mean) ** 2, 0);
                    if (sst > 0) add('r2', 1 - sse / sst);
                    add('rmse', Math.sqrt(sse / n));
                    add('mae', yy.reduce((a, v, i) => a + Math.abs(v - pp[i]), 0) / n);
                }
            }
            const out = { B, method: 'percentile bootstrap of test patients' };
            Object.keys(stats).forEach(k => {
                const v = stats[k].sort((a, b) => a - b);
                const q = f => v[Math.min(v.length - 1, Math.max(0, Math.floor(f * (v.length - 1))))];
                if (v.length >= 20) out[k] = [Number(q(0.025).toFixed(4)), Number(q(0.975).toFixed(4))];
            });
            return out;
        },

        /**
         * Logistic regression fitted to convergence by iteratively reweighted least squares (Newton), for
         * explanation: coefficients, standard errors, Wald 95% CIs, p-values and odds ratios. Binary outcomes.
         * X should not contain a constant column; an intercept is added. Returns null (with a note) when the
         * fit does not converge (e.g. complete separation) or the design is singular.
         */
        logisticIRLS(X, y, names = null, maxIter = 100) {
            const n = X.length;
            if (!n) return null;
            const p = X[0].length + 1;
            const Z = X.map(r => [1, ...r]);
            let beta = new Array(p).fill(0);
            let converged = false, info = null;
            for (let it = 0; it < maxIter; it++) {
                const H = Array.from({ length: p }, () => new Array(p).fill(0));
                const g = new Array(p).fill(0);
                for (let i = 0; i < n; i++) {
                    const eta = Z[i].reduce((a, v, j) => a + v * beta[j], 0);
                    const mu = 1 / (1 + Math.exp(-eta));
                    const w = Math.max(mu * (1 - mu), 1e-12);
                    for (let j = 0; j < p; j++) {
                        g[j] += (y[i] - mu) * Z[i][j];
                        for (let k = j; k < p; k++) H[j][k] += w * Z[i][j] * Z[i][k];
                    }
                }
                for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) H[j][k] = H[k][j];
                info = H;
                const step = solveLinear(H, g);
                if (!step) return { converged: false, note: 'The predictors are collinear (singular design), so odds ratios cannot be estimated.' };
                beta = beta.map((b, j) => b + step[j]);
                if (beta.some(b => !isFinite(b) || Math.abs(b) > 50)) return { converged: false, note: 'The model did not converge: some predictors separate the outcomes (almost) perfectly, so odds ratios are not estimable.' };
                if (Math.max(...step.map(Math.abs)) < 1e-9) { converged = true; break; }
            }
            if (!converged) return { converged: false, note: 'The model did not converge within the iteration limit.' };
            const inv = invertMatrix(info);
            if (!inv) return { converged: false, note: 'The information matrix is singular, so standard errors cannot be estimated.' };
            const rows = beta.map((b, j) => {
                const se = Math.sqrt(Math.max(inv[j][j], 0));
                const z = se > 0 ? b / se : null;
                return {
                    name: j === 0 ? '(Intercept)' : (names ? names[j - 1] : `x${j}`),
                    coef: b, se, z, p: z === null ? null : 2 * (1 - normalCdf(Math.abs(z))),
                    or: Math.exp(b), orLow: Math.exp(b - 1.959964 * se), orHigh: Math.exp(b + 1.959964 * se)
                };
            });
            return { converged: true, n, events: y.reduce((a, v) => a + v, 0), rows };
        },

        // ---------------------------------------------------------------------------------------------
        // Saving and loading trained models (JSON). Arrays only; TF.js networks are stored as topology +
        // base64 weights. k-NN models keep their (scaled) training cases, which is flagged on export.
        // ---------------------------------------------------------------------------------------------
        async serializeModel(model) {
            if (!model) return null;
            const plain = (obj, skip = []) => {
                const o = {};
                Object.keys(obj).forEach(k => {
                    if (skip.includes(k) || k === 'rng' || typeof obj[k] === 'function') return;
                    const v = obj[k];
                    o[k] = ArrayBuffer.isView(v) ? { __typed: v.constructor.name, data: Array.from(v) } : v;
                });
                return o;
            };
            if (model instanceof RandomForest) {
                return { type: 'rf', state: plain(model, ['trees']), trees: model.trees.map(t => ({ state: plain(t, ['root']), root: t.root })) };
            }
            if (model instanceof GeneralizedLinearModel) return { type: 'glm', state: plain(model) };
            if (model instanceof KNNModel) return { type: 'knn', state: plain(model) };
            if (model instanceof GaussianNaiveBayes) return { type: 'nb', state: plain(model) };
            if (model instanceof GradientBoostingModel) return { type: 'gbm', state: plain(model) };
            if (model instanceof DeepNeuralNetwork) {
                const out = { type: 'nn', state: plain(model, ['tfModel', 'fallbackGLM', 'history']) };
                if (model.fallbackGLM) out.fallbackGLM = await this.serializeModel(model.fallbackGLM);
                if (model.tfModel) {
                    let artifacts = null;
                    await model.tfModel.save(tf.io.withSaveHandler(async a => { artifacts = a; return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } }; }));
                    const bytes = new Uint8Array(artifacts.weightData);
                    let bin = '';
                    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
                    out.tf = { modelTopology: artifacts.modelTopology, weightSpecs: artifacts.weightSpecs, weightData: btoa(bin) };
                }
                return out;
            }
            if (model instanceof CalibratedModel) {
                return { type: 'calibrated', calibrator: model.calibrator, base: await this.serializeModel(model.base) };
            }
            if (model instanceof EnsembleModel) {
                const items = [];
                for (const it of model.models) items.push({ ...it, model: await this.serializeModel(it.model) });
                return { type: 'ensemble', state: plain(model, ['models']), models: items };
            }
            throw new Error('This model type cannot be exported.');
        },

        async deserializeModel(obj) {
            if (!obj) return null;
            const restore = (proto, state) => {
                const m = Object.create(proto);
                Object.keys(state || {}).forEach(k => {
                    const v = state[k];
                    m[k] = v && v.__typed ? new globalThis[v.__typed](v.data) : v;
                });
                m.rng = createPRNG(m.seed);
                return m;
            };
            if (obj.type === 'rf') {
                const m = restore(RandomForest.prototype, obj.state);
                m.trees = obj.trees.map(t => { const tr = restore(DecisionTree.prototype, t.state); tr.root = t.root; return tr; });
                return m;
            }
            if (obj.type === 'glm') return restore(GeneralizedLinearModel.prototype, obj.state);
            if (obj.type === 'knn') return restore(KNNModel.prototype, obj.state);
            if (obj.type === 'nb') return restore(GaussianNaiveBayes.prototype, obj.state);
            if (obj.type === 'gbm') return restore(GradientBoostingModel.prototype, obj.state);
            if (obj.type === 'nn') {
                const m = restore(DeepNeuralNetwork.prototype, obj.state);
                m.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
                m.fallbackGLM = obj.fallbackGLM ? await this.deserializeModel(obj.fallbackGLM) : null;
                m.tfModel = null;
                if (obj.tf) {
                    if (typeof tf === 'undefined') throw new Error('TensorFlow.js is needed to load this neural network.');
                    const bin = atob(obj.tf.weightData);
                    const bytes = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                    m.tfModel = await tf.loadLayersModel(tf.io.fromMemory({ modelTopology: obj.tf.modelTopology, weightSpecs: obj.tf.weightSpecs, weightData: bytes.buffer }));
                }
                return m;
            }
            if (obj.type === 'calibrated') {
                return new CalibratedModel(await this.deserializeModel(obj.base), obj.calibrator);
            }
            if (obj.type === 'ensemble') {
                const m = restore(EnsembleModel.prototype, obj.state);
                m.models = [];
                for (const it of obj.models) m.models.push({ ...it, model: await this.deserializeModel(it.model) });
                return m;
            }
            throw new Error(`Unknown model type in file: ${obj.type}`);
        },

        /**
         * Minimum sample size for developing a prediction model (Riley et al., BMJ 2020; Stat Med 2019), as in the
         * pmsampsize package. Binary: p = candidate parameters, prevalence, anticipated Cox-Snell R² (default 15% of
         * its maximum). Continuous: p and anticipated R². Returns each criterion and the overall minimum.
         */
        rileySampleSize({ type, p, prevalence = null, r2 = null, shrinkage = 0.9, optimism = 0.05, riskMargin = 0.05 }) {
            const crit = [];
            const nShrink = (R2) => Math.ceil(p / ((shrinkage - 1) * Math.log(1 - R2 / shrinkage)));
            if (type === 'binary') {
                if (!(prevalence > 0 && prevalence < 1)) return null;
                const maxR2 = 1 - Math.pow(Math.pow(prevalence, prevalence) * Math.pow(1 - prevalence, 1 - prevalence), 2);
                const R2 = r2 !== null && r2 > 0 ? Math.min(r2, maxR2 * 0.999) : 0.15 * maxR2;
                const S2 = R2 / (R2 + optimism * maxR2);
                crit.push({ id: 'shrinkage', label: `Shrinkage ≥ ${shrinkage} (limited overfitting)`, n: nShrink(R2) });
                crit.push({ id: 'optimism', label: `Optimism in Nagelkerke R² ≤ ${optimism}`, n: Math.ceil(p / ((S2 - 1) * Math.log(1 - R2 / S2))) });
                crit.push({ id: 'risk', label: `Overall risk estimated within ±${riskMargin}`, n: Math.ceil(Math.pow(1.959964 / riskMargin, 2) * prevalence * (1 - prevalence)) });
                const n = Math.max(...crit.map(c => c.n));
                return { type, p, prevalence, r2: R2, maxR2, criteria: crit, n, events: Math.ceil(n * prevalence), epp: (n * prevalence) / p, r2Assumed: !(r2 !== null && r2 > 0) };
            }
            const R2 = r2 !== null && r2 > 0 && r2 < 1 ? r2 : 0.2;
            crit.push({ id: 'shrinkage', label: `Shrinkage ≥ ${shrinkage} (limited overfitting)`, n: nShrink(R2) });
            crit.push({ id: 'optimism', label: `Optimism in adjusted R² ≤ ${optimism}`, n: Math.ceil(1 + (p * (1 - R2)) / optimism) });
            crit.push({ id: 'residual', label: 'Residual SD estimated within a 10% margin', n: 234 + p });
            return { type, p, r2: R2, criteria: crit, n: Math.max(...crit.map(c => c.n)), r2Assumed: !(r2 !== null && r2 > 0 && r2 < 1) };
        },

        tCdf(t, df) { return tCdf(t, df); },
        tQuantile(p, df) { return tQuantile(p, df); },

        /**
         * Rubin's rules: pools m estimates (each a vector of coefficients with their standard errors) from
         * multiply imputed datasets. Returns pooled coefficient, total SE, degrees of freedom (Barnard–Rubin
         * small-sample adjustment with the complete-data df), 95% CI, p-value and the fraction of missing information.
         */
        poolRubin(estimates, completeDf = Infinity) {
            const m = estimates.length;
            if (m < 2) return null;
            const k = estimates[0].coef.length;
            const out = [];
            for (let j = 0; j < k; j++) {
                const q = estimates.map(e => e.coef[j]);
                const u = estimates.map(e => e.se[j] * e.se[j]);
                const qbar = q.reduce((a, b) => a + b, 0) / m;
                const W = u.reduce((a, b) => a + b, 0) / m;
                const Bv = q.reduce((a, b) => a + (b - qbar) ** 2, 0) / (m - 1);
                const T = W + (1 + 1 / m) * Bv;
                const lambda = T > 0 ? ((1 + 1 / m) * Bv) / T : 0;
                const dfOld = lambda > 0 ? (m - 1) / (lambda * lambda) : Infinity;
                let df = dfOld;
                if (isFinite(completeDf)) {
                    const dfObs = ((completeDf + 1) / (completeDf + 3)) * completeDf * (1 - lambda);
                    df = isFinite(dfOld) ? (dfOld * dfObs) / (dfOld + dfObs) : dfObs;
                }
                const se = Math.sqrt(T);
                const tq = tQuantile(0.975, df);
                const tStat = se > 0 ? qbar / se : null;
                out.push({ coef: qbar, se, df, low: qbar - tq * se, high: qbar + tq * se, p: tStat === null ? null : 2 * (1 - tCdf(Math.abs(tStat), df)), fmi: lambda, within: W, between: Bv });
            }
            return out;
        },

        /** Harrell's concordance index: share of comparable pairs (earlier event vs later time) ranked correctly by risk */
        concordanceIndex(time, event, risk) {
            let conc = 0, comp = 0;
            const n = time.length;
            for (let i = 0; i < n; i++) {
                if (event[i] !== 1) continue;
                for (let j = 0; j < n; j++) {
                    if (time[j] > time[i] || (time[j] === time[i] && event[j] === 0 && j !== i)) {
                        comp++;
                        if (risk[i] > risk[j]) conc++; else if (risk[i] === risk[j]) conc += 0.5;
                    }
                }
            }
            return comp ? conc / comp : null;
        },

        /** Kaplan-Meier survival estimate: [{time, survival, atRisk, events}] */
        kaplanMeier(time, event) {
            const ts = [...new Set(time)].sort((a, b) => a - b);
            let S = 1;
            const out = [{ time: 0, survival: 1, atRisk: time.length, events: 0 }];
            ts.forEach(t => {
                const atRisk = time.filter(v => v >= t).length;
                const d = time.filter((v, i) => v === t && event[i] === 1).length;
                if (d > 0 && atRisk > 0) { S *= 1 - d / atRisk; out.push({ time: t, survival: S, atRisk, events: d }); }
            });
            return out;
        },
        kmAt(km, t) { let s = 1; for (const k of km) { if (k.time <= t) s = k.survival; else break; } return s; },
        /** Kaplan-Meier value just before t (left limit), as needed for censoring weights */
        kmBefore(km, t) { let s = 1; for (const k of km) { if (k.time < t) s = k.survival; else break; } return s; },

        /**
         * Censoring distribution G(t) = P(still uncensored at t): the Kaplan-Meier estimate with censoring as the
         * "event" (from the training patients). Used to weight the time-dependent measures (IPCW).
         */
        censoringKM(time, event) { return this.kaplanMeier(time, event.map(e => (e === 1 ? 0 : 1))); },

        /**
         * Time-dependent AUC at horizon tau (cumulative cases / dynamic controls, Uno et al. 2007): of a
         * patient with the event by tau and one still event-free after tau, how often the first had the
         * higher risk. Cases are weighted by 1/G(T−) for censoring. Returns null if not estimable.
         */
        timeDependentAUC(time, event, risk, tau, censKM) {
            let num = 0, den = 0;
            const controls = [];
            for (let j = 0; j < time.length; j++) if (time[j] > tau) controls.push(risk[j]);
            if (!controls.length) return null;
            for (let i = 0; i < time.length; i++) {
                if (!(event[i] === 1 && time[i] <= tau)) continue;
                const g = this.kmBefore(censKM, time[i]);
                if (!(g > 0)) continue;
                const w = 1 / g;
                let c = 0;
                controls.forEach(r => { if (risk[i] > r) c += 1; else if (risk[i] === r) c += 0.5; });
                num += w * c;
                den += w * controls.length;
            }
            return den > 0 ? num / den : null;
        },

        /**
         * Brier score at horizon tau with censoring weights (Graf et al. 1999): squared error of the predicted
         * survival probability S(tau | x) against the observed status. Patients censored before tau get weight 0
         * and the others are up-weighted to represent them.
         */
        ipcwBrier(time, event, survAtTau, tau, censKM) {
            const gTau = this.kmAt(censKM, tau);
            let sum = 0;
            for (let i = 0; i < time.length; i++) {
                if (time[i] <= tau && event[i] === 1) { const g = this.kmBefore(censKM, time[i]); if (g > 0) sum += (survAtTau[i] ** 2) / g; }
                else if (time[i] > tau && gTau > 0) sum += ((1 - survAtTau[i]) ** 2) / gTau;
            }
            return time.length ? sum / time.length : null;
        },
        createCox(config) { return new CoxModel(config); },

        /** Platt scaling: logistic regression of the outcome on logit(p) (Newton-Raphson); null if not estimable */
        fitPlatt(y, p) {
            const lp = p.map(logitClip);
            const fit = this.logisticIRLS(lp.map(v => [v]), y.map(v => (v === 1 ? 1 : 0)));
            if (!fit || !fit.converged) return null;
            return { type: 'platt', a: fit.rows[0].coef, b: fit.rows[1].coef, n: y.length };
        },

        /** Isotonic regression (pool-adjacent-violators) of the outcome on p: a non-decreasing step function */
        fitIsotonic(y, p) {
            const idx = p.map((v, i) => i).sort((a, b) => p[a] - p[b]);
            const blocks = []; // {sumY, w, xMin, xMax}
            idx.forEach(i => {
                blocks.push({ sumY: y[i] === 1 ? 1 : 0, w: 1, xMin: p[i], xMax: p[i] });
                while (blocks.length > 1 && blocks[blocks.length - 2].sumY / blocks[blocks.length - 2].w >= blocks[blocks.length - 1].sumY / blocks[blocks.length - 1].w) {
                    const b2 = blocks.pop(), b1 = blocks.pop();
                    blocks.push({ sumY: b1.sumY + b2.sumY, w: b1.w + b2.w, xMin: b1.xMin, xMax: b2.xMax });
                }
            });
            const x = [], yv = [];
            blocks.forEach(b => { const v = b.sumY / b.w; x.push(b.xMin); yv.push(v); if (b.xMax > b.xMin) { x.push(b.xMax); yv.push(v); } });
            return { type: 'isotonic', x, y: yv, n: y.length };
        },

        wrapCalibrated(model, calibrator) { return new CalibratedModel(model, calibrator); },
        applyCalibrator(cal, p) { return applyCalibrator(cal, p); },

        /** Re-labels one binary output with a decision threshold on the positive-class probability */
        applyDecisionThreshold(o, threshold) {
            if (!o || !o.probabilities) return o;
            const p1 = Number(o.probabilities[1]);
            if (!isFinite(p1)) return o;
            const pred = p1 >= threshold ? 1 : 0;
            const p = Number(o.probabilities[pred]);
            return { ...o, pred, confidence: isFinite(p) ? p : o.confidence };
        },

        /**
         * Calibration of binary probabilities on a held-out set: grouped observed-vs-predicted points for a
         * calibration plot, the calibration slope (logistic regression of the outcome on logit(p); ideal 1) and
         * the calibration intercept / calibration-in-the-large (logistic regression with logit(p) as an offset;
         * ideal 0). Slope and intercept are left empty when they cannot be estimated (e.g. perfect separation).
         */
        calibrationSummary(y, p) {
            const n = y.length;
            const events = y.reduce((a, b) => a + b, 0);
            const out = { n, events, bins: [], slope: null, intercept: null, note: null };
            if (n === 0) return out;
            // Groups of roughly equal size, ordered by predicted probability
            const nBins = n >= 10 ? Math.max(2, Math.min(10, Math.floor(n / 10))) : 0;
            if (nBins) {
                const order = p.map((v, i) => i).sort((a, b) => p[a] - p[b]);
                for (let b = 0; b < nBins; b++) {
                    const idx = order.slice(Math.round((b * n) / nBins), Math.round(((b + 1) * n) / nBins));
                    if (!idx.length) continue;
                    out.bins.push({
                        meanPred: Number((idx.reduce((a, i) => a + p[i], 0) / idx.length).toFixed(4)),
                        observed: Number((idx.reduce((a, i) => a + y[i], 0) / idx.length).toFixed(4)),
                        n: idx.length
                    });
                }
            }
            if (events === 0 || events === n || n < 10) {
                out.note = n < 10 ? 'Too few test cases to estimate calibration (fewer than 10).' : 'Only one outcome present in the test set: calibration cannot be estimated.';
                return out;
            }
            const eps = 1e-4;
            const lp = p.map(v => { const q = Math.min(1 - eps, Math.max(eps, Number(v) || 0)); return Math.log(q / (1 - q)); });
            const sig = z => 1 / (1 + Math.exp(-z));
            // Slope: y ~ a + b·lp (Newton-Raphson)
            let a = 0, b = 1, ok = false;
            for (let it = 0; it < 100; it++) {
                let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
                for (let i = 0; i < n; i++) {
                    const m = sig(a + b * lp[i]);
                    const w = m * (1 - m);
                    g0 += y[i] - m; g1 += (y[i] - m) * lp[i];
                    h00 += w; h01 += w * lp[i]; h11 += w * lp[i] * lp[i];
                }
                const det = h00 * h11 - h01 * h01;
                if (!isFinite(det) || Math.abs(det) < 1e-12) break;
                const da = (h11 * g0 - h01 * g1) / det, db = (h00 * g1 - h01 * g0) / det;
                a += da; b += db;
                if (!isFinite(a) || !isFinite(b) || Math.abs(b) > 100) break;
                if (Math.abs(da) < 1e-8 && Math.abs(db) < 1e-8) { ok = true; break; }
            }
            if (ok) out.slope = Number(b.toFixed(3));
            // Intercept (calibration-in-the-large): y ~ a + offset(lp)
            let c = 0, okC = false;
            for (let it = 0; it < 100; it++) {
                let g = 0, h = 0;
                for (let i = 0; i < n; i++) { const m = sig(c + lp[i]); g += y[i] - m; h += m * (1 - m); }
                if (h < 1e-12) break;
                const dc = g / h; c += dc;
                if (!isFinite(c) || Math.abs(c) > 100) break;
                if (Math.abs(dc) < 1e-10) { okC = true; break; }
            }
            if (okC && ok) out.intercept = Number(c.toFixed(3)); // under (near-)separation neither is meaningful
            if (!ok || !okC) out.note = 'Calibration slope/intercept could not be estimated: the predictions separate the outcomes (almost) perfectly on these test cases, or there are too few events.';
            return out;
        },

        _calculateBinaryAUC(yTrue, yScores) {
            // Rank-based Mann-Whitney U test calculation for ROC AUC
            const items = yTrue.map((val, idx) => ({ y: val, score: yScores[idx] }));
            items.sort((a, b) => a.score - b.score);

            let nPos = 0;
            let nNeg = 0;
            items.forEach(it => {
                if (it.y === 1) nPos++;
                else nNeg++;
            });

            // AUC is undefined when the test set contains only one class
            if (nPos === 0 || nNeg === 0) return null;

            // Mid-ranks for tied scores (ties are common with vote-based probabilities)
            let rankSumPos = 0;
            let i = 0;
            while (i < items.length) {
                let j = i;
                while (j + 1 < items.length && items[j + 1].score === items[i].score) j++;
                const avgRank = (i + j + 2) / 2;
                for (let k = i; k <= j; k++) {
                    if (items[k].y === 1) rankSumPos += avgRank;
                }
                i = j + 1;
            }

            const u = rankSumPos - (nPos * (nPos + 1)) / 2;
            const auc = u / (nPos * nNeg);
            return Number(Math.max(0, Math.min(1, auc)).toFixed(4));
        }
    };

    // Background-tab-safe yield, shared with app.js (e.g. automatic tuning loops)
    ModelEngine.yieldToBrowser = yieldToBrowser;
    window.ModelEngine = ModelEngine;
    window.ModelFactory = ModelEngine;
    window.RandomForest = RandomForest;
    window.KNNModel = KNNModel;
    window.GeneralizedLinearModel = GeneralizedLinearModel;
    window.DeepNeuralNetwork = DeepNeuralNetwork;
    window.GaussianNaiveBayes = GaussianNaiveBayes;
    window.EnsembleModel = EnsembleModel;
    window.CalibratedModel = CalibratedModel;
    window.GradientBoostingModel = GradientBoostingModel;
    window.CoxModel = CoxModel;
})(window);
