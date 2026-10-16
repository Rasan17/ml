// training-setup.js: part of the Universal Medical ML/NeuralNet Predictor application controller
import { AppState, MODEL_INFO, parseSeed } from './core.js';
import { DOM } from './shell.js';
import { showSeedToast } from './training-actions.js';
// =========================================================================
// STEP 4: MODEL TRAINING & EVALUATION
// =========================================================================
function updateHyperparamsVisibility(targetModelId) {
    const activeId = targetModelId || AppState.activeHyperparamModelId || AppState.selectedModelId || 'nn';
    AppState.activeHyperparamModelId = activeId;
    const isNN = activeId === 'nn';
    const isRF = activeId === 'rf';
    const isGLM = activeId === 'glm';
    const isKNN = activeId === 'knn';
    const isNB = activeId === 'nb';
    const isEnsemble = activeId === 'ensemble';
    const gbmBox = document.getElementById('gbm-params-box');
    if (gbmBox) gbmBox.style.display = activeId === 'gbm' ? 'grid' : 'none';

    const nnBox = document.getElementById('nn-params-box');
    const rfBox = document.getElementById('rf-params-box');
    const glmBox = document.getElementById('glm-params-box');
    const knnBox = document.getElementById('knn-params-box');
    const nbBox = document.getElementById('nb-params-box');
    const ensembleBox = document.getElementById('ensemble-params-box');

    if (nnBox) nnBox.style.display = isNN ? 'grid' : 'none';
    if (rfBox) rfBox.style.display = isRF ? 'grid' : 'none';
    if (glmBox) glmBox.style.display = isGLM ? 'grid' : 'none';
    if (knnBox) knnBox.style.display = isKNN ? 'grid' : 'none';
    if (nbBox) nbBox.style.display = isNB ? 'grid' : 'none';
    if (ensembleBox) ensembleBox.style.display = isEnsemble ? 'grid' : 'none';

    if (DOM.activeHyperparamSelect && DOM.activeHyperparamSelect.value !== activeId) {
        DOM.activeHyperparamSelect.value = activeId;
    }
}

// =========================================================================
// REPRODUCIBILITY SEED MANAGEMENT (DEFAULT 42 VS CUSTOM SEED)
// =========================================================================
function setupSeedManagement() {
    const ALL_MODELS = ['nn', 'rf', 'glm', 'knn', 'nb', 'ensemble'];

    // Helper to update a model's seed mode (default 42 vs custom)
    function setModelSeedMode(mId, mode) {
        const cardDefaultRadio = document.querySelector(`input[name="seed-mode-${mId}"][value="default"]`);
        const cardCustomRadio = document.querySelector(`input[name="seed-mode-${mId}"][value="custom"]`);
        const hpDefaultRadio = document.querySelector(`input[name="hp-seed-mode-${mId}"][value="default"]`);
        const hpCustomRadio = document.querySelector(`input[name="hp-seed-mode-${mId}"][value="custom"]`);
        const cardCustomRow = document.getElementById(`custom-seed-row-${mId}`);
        const hpCustomWrap = document.getElementById(`hp-seed-wrap-${mId}`);
        const cardInput = document.getElementById(`${mId}-card-seed-input`);
        const hpInput = document.getElementById(`${mId}-seed-input`);

        if (mode === 'default') {
            if (cardDefaultRadio) cardDefaultRadio.checked = true;
            if (hpDefaultRadio) hpDefaultRadio.checked = true;
            if (cardCustomRow) cardCustomRow.style.display = 'none';
            if (hpCustomWrap) hpCustomWrap.style.display = 'none';
            if (cardInput) cardInput.value = 42;
            if (hpInput) hpInput.value = 42;
        } else {
            if (cardCustomRadio) cardCustomRadio.checked = true;
            if (hpCustomRadio) hpCustomRadio.checked = true;
            if (cardCustomRow) cardCustomRow.style.display = 'flex';
            if (hpCustomWrap) hpCustomWrap.style.display = 'flex';
            // "Custom" still showing 42 would be indistinguishable from Default: propose a random seed
            const currentVal = (cardInput && cardInput.value) || (hpInput && hpInput.value) || '42';
            if (String(currentVal).trim() === '42') {
                const newSeed = Math.floor(Math.random() * 90000) + 10000;
                if (cardInput) cardInput.value = newSeed;
                if (hpInput) hpInput.value = newSeed;
                showSeedToast(`Custom seed for ${MODEL_INFO[mId]?.name || mId}: ${newSeed} (edit if required)`);
            }
            if (cardInput && (!cardInput.value || cardInput.value === '42')) cardInput.focus();
            else if (hpInput) hpInput.focus();
        }
    }

    // Attach listeners for Card Seed Radios and Hyperparam Box Seed Radios
    ALL_MODELS.forEach(mId => {
        const cardRadios = document.querySelectorAll(`input[name="seed-mode-${mId}"]`);
        cardRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                setModelSeedMode(mId, e.target.value);
            });
        });

        const hpRadios = document.querySelectorAll(`input[name="hp-seed-mode-${mId}"]`);
        hpRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                setModelSeedMode(mId, e.target.value);
            });
        });

        // Two-way sync for custom seed inputs
        const cardInput = document.getElementById(`${mId}-card-seed-input`);
        const hpInput = document.getElementById(`${mId}-seed-input`);

        if (cardInput && hpInput) {
            cardInput.addEventListener('input', (e) => {
                hpInput.value = e.target.value;
            });
            hpInput.addEventListener('input', (e) => {
                cardInput.value = e.target.value;
            });
        }

        // Dice roll buttons for Card
        const cardDice = document.querySelector(`.card-btn-roll-seed[data-model="${mId}"]`);
        if (cardDice) {
            cardDice.addEventListener('click', (e) => {
                e.stopPropagation();
                const newSeed = Math.floor(Math.random() * 90000) + 10000;
                if (cardInput) cardInput.value = newSeed;
                if (hpInput) hpInput.value = newSeed;
                showSeedToast(`Generated ${MODEL_INFO[mId]?.name || mId} Seed: ${newSeed}`);
            });
        }

        // Dice roll buttons for Hyperparam Box
        const hpDice = document.getElementById(`btn-roll-${mId}-seed`);
        if (hpDice) {
            hpDice.addEventListener('click', (e) => {
                e.stopPropagation();
                const newSeed = Math.floor(Math.random() * 90000) + 10000;
                if (cardInput) cardInput.value = newSeed;
                if (hpInput) hpInput.value = newSeed;
                showSeedToast(`Generated ${MODEL_INFO[mId]?.name || mId} Seed: ${newSeed}`);
            });
        }
    });

    // Split Seed (Option A) Controls
    const splitRadios = document.querySelectorAll('input[name="split-seed-mode"]');
    const splitCustomWrap = document.getElementById('split-seed-custom-wrap');
    const splitInput = document.getElementById('split-seed-input');
    const splitDice = document.getElementById('btn-roll-split-seed');

    splitRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'default') {
                if (splitCustomWrap) splitCustomWrap.style.display = 'none';
                if (splitInput) splitInput.value = 42;
            } else {
                if (splitCustomWrap) splitCustomWrap.style.display = 'flex';
                if (splitInput && String(splitInput.value).trim() === '42') {
                    splitInput.value = Math.floor(Math.random() * 90000) + 10000;
                    showSeedToast(`Custom split seed: ${splitInput.value} (edit if required)`);
                }
                if (splitInput) splitInput.focus();
            }
        });
    });

    if (splitDice && splitInput) {
        splitDice.addEventListener('click', () => {
            const newSeed = Math.floor(Math.random() * 90000) + 10000;
            splitInput.value = newSeed;
            splitInput.focus();
            showSeedToast(`Generated Split Seed: ${newSeed}`);
        });
    }

    // Quick Global Seed Action Buttons
    const btnAllDefault = document.getElementById('btn-all-default-seed');
    if (btnAllDefault) {
        btnAllDefault.addEventListener('click', () => {
            ALL_MODELS.forEach(mId => setModelSeedMode(mId, 'default'));
            const defSplitRadio = document.getElementById('split-seed-mode-default');
            if (defSplitRadio) {
                defSplitRadio.checked = true;
                if (splitCustomWrap) splitCustomWrap.style.display = 'none';
                if (splitInput) splitInput.value = 42;
            }
            showSeedToast('All models set to Default Seed (42)');
        });
    }

    const btnAllRandom = document.getElementById('btn-all-random-seed');
    if (btnAllRandom) {
        btnAllRandom.addEventListener('click', () => {
            ALL_MODELS.forEach(mId => {
                setModelSeedMode(mId, 'custom');
                const randSeed = Math.floor(Math.random() * 90000) + 10000;
                const cInp = document.getElementById(`${mId}-card-seed-input`);
                const hInp = document.getElementById(`${mId}-seed-input`);
                if (cInp) cInp.value = randSeed;
                if (hInp) hInp.value = randSeed;
            });
            showSeedToast('All models assigned individual random seeds 🎲');
        });
    }
}

function getHyperparamsForModel(modelId, prep) {
    const hp = {};
    if (modelId === 'nn') {
        hp.epochs = parseInt(document.getElementById('nn-epochs-input')?.value, 10) || 50;
        hp.learningRate = parseFloat(document.getElementById('nn-lr-input')?.value) || 0.005;
        const h1 = parseInt(document.getElementById('nn-h1-input')?.value, 10) || 64;
        const h2 = parseInt(document.getElementById('nn-h2-input')?.value, 10) || 32;
        const h3El = document.getElementById('nn-h3-input');
        const h3 = h3El ? (parseInt(h3El.value, 10) || 0) : 0;
        hp.hiddenLayers = h3 > 0 ? [h1, h2, h3] : [h1, h2];
        const dropoutVal = parseFloat(document.getElementById('nn-dropout-input')?.value);
        hp.dropoutRate = isNaN(dropoutVal) ? 0.1 : Math.min(0.9, Math.max(0, dropoutVal)); // 0 is a valid choice
        hp.activation = document.getElementById('nn-act-select')?.value || 'relu';
        const optEl = document.getElementById('nn-optimizer-select');
        hp.optimizer = optEl ? optEl.value : 'adam';
        const batchEl = document.getElementById('nn-batch-select');
        hp.batchSize = batchEl ? (parseInt(batchEl.value, 10) || 16) : 16;
        const l2El = document.getElementById('nn-l2-select');
        hp.l2Reg = l2El ? parseFloat(l2El.value) : 0.0001;
    } else if (modelId === 'rf') {
        hp.numTrees = parseInt(document.getElementById('rf-trees-input')?.value, 10) || 30;
        hp.maxDepth = parseInt(document.getElementById('rf-depth-input')?.value, 10) || 6;
        const minSplitEl = document.getElementById('rf-minsplit-select');
        hp.minSamplesSplit = minSplitEl ? (parseInt(minSplitEl.value, 10) || 2) : 2;
        const maxFeatEl = document.getElementById('rf-maxfeatures-select');
        hp.featureRatio = maxFeatEl ? maxFeatEl.value : 'sqrt';
        hp.splitMode = document.getElementById('rf-split-select')?.value || 'auto';
    } else if (modelId === 'glm') {
        hp.lr = parseFloat(document.getElementById('glm-lr-input')?.value) || 0.05;
        hp.epochs = parseInt(document.getElementById('glm-epochs-input')?.value, 10) || 150;
        const lambdaEl = document.getElementById('glm-lambda-select');
        hp.lambdaL2 = lambdaEl ? parseFloat(lambdaEl.value) : 0.01;
        const solverEl = document.getElementById('glm-solver-select');
        hp.solver = solverEl ? solverEl.value : 'momentum';
    } else if (modelId === 'knn') {
        hp.k = parseInt(document.getElementById('knn-k-input')?.value, 10) || 5;
        const metricEl = document.getElementById('knn-metric-select');
        hp.metric = metricEl ? metricEl.value : 'euclidean';
        const weightEl = document.getElementById('knn-weighting-select');
        hp.weighting = weightEl ? weightEl.value : 'uniform';
    } else if (modelId === 'gbm') {
        hp.nEstimators = parseInt(document.getElementById('gbm-rounds-input')?.value, 10) || 200;
        hp.learningRate = parseFloat(document.getElementById('gbm-lr-select')?.value) || 0.1;
        hp.maxDepth = parseInt(document.getElementById('gbm-depth-input')?.value, 10) || 3;
        hp.minSamplesLeaf = parseInt(document.getElementById('gbm-minleaf-select')?.value, 10) || 5;
        const l2 = parseFloat(document.getElementById('gbm-l2-select')?.value);
        hp.l2 = isNaN(l2) ? 1 : l2;
        hp.subsample = parseFloat(document.getElementById('gbm-subsample-select')?.value) || 0.8;
    } else if (modelId === 'nb') {
        const smoothEl = document.getElementById('nb-smoothing-select');
        hp.varSmoothing = smoothEl ? parseFloat(smoothEl.value) : 1e-4;
        const priorsEl = document.getElementById('nb-priors-select');
        hp.priorMode = priorsEl ? priorsEl.value : 'empirical';
    } else if (modelId === 'ensemble') {
        const selectedBases = [];
        if (document.getElementById('ens-include-nn')?.checked) selectedBases.push('nn');
        if (document.getElementById('ens-include-rf')?.checked) selectedBases.push('rf');
        if (document.getElementById('ens-include-glm')?.checked) selectedBases.push('glm');
        if (document.getElementById('ens-include-knn')?.checked) selectedBases.push('knn');
        if (document.getElementById('ens-include-nb')?.checked && prep?.isClassification) selectedBases.push('nb');
        if (selectedBases.length === 0) selectedBases.push('rf', 'glm');

        hp.voting = document.getElementById('ensemble-voting-select')?.value || 'soft';
        hp.baseModels = selectedBases;
        const powerEl = document.getElementById('ens-power-select');
        hp.power = powerEl ? parseFloat(powerEl.value) : 2;
        const capEl = document.getElementById('ens-capacity-select');
        hp.capacity = capEl ? capEl.value : 'balanced';
    }

    // Extract Reproducibility Seed for this model (Default 42 vs Custom Seed)
    const cardModeRadio = document.querySelector(`input[name="seed-mode-${modelId}"]:checked`);
    const hpModeRadio = document.querySelector(`input[name="hp-seed-mode-${modelId}"]:checked`);
    const isDefault = (cardModeRadio ? cardModeRadio.value === 'default' : true) &&
                      (hpModeRadio ? hpModeRadio.value === 'default' : true);

    let seedVal = 42;
    if (!isDefault) {
        const cardInput = document.getElementById(`${modelId}-card-seed-input`);
        const hpInput = document.getElementById(`${modelId}-seed-input`);
        const inputVal = (cardInput && cardInput.value !== '') ? cardInput.value : (hpInput ? hpInput.value : '');
        const parsed = parseSeed(inputVal);
        seedVal = parsed !== null ? parsed : 42;
        // Keep the on-screen seed identical to the seed actually used
        if (parsed === null) {
            if (cardInput) cardInput.value = 42;
            if (hpInput) hpInput.value = 42;
        }
    }
    hp.seed = seedVal;

    return hp;
}

export { updateHyperparamsVisibility, setupSeedManagement, getHyperparamsForModel };
