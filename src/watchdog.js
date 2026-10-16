// watchdog.js: part of the Universal Medical ML/NeuralNet Predictor application controller



// =========================================================================
// EXECUTION WATCHDOG: LIVE MONITOR, LOOP DETECTOR & STALL SENTINEL
// =========================================================================
class ExecutionWatchdog {
    constructor(config) {
        this.container = config.container;
        this.badge = config.badge;
        this.stateText = config.stateText;
        this.timerText = config.timerText;
        this.loopText = config.loopText;
        this.loopStatus = config.loopStatus;
        this.taskName = config.taskName;
        this.stepCounter = config.stepCounter;
        this.progressBar = config.progressBar;
        this.threadStatus = config.threadStatus;
        this.heartbeatTick = config.heartbeatTick;
        this.stuckHint = config.stuckHint;
        this.cancelBtn = config.cancelBtn;

        this.startTime = 0;
        this.lastTickTime = 0;
        this.timerInterval = null;
        this.aborted = false;
        this.transitionHistory = [];
        this.loopCount = 0;
        this.onCancelCallback = null;

        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.forceAbort());
        }
    }

    start(taskDescription, totalSteps = 1, onCancel = null) {
        this.aborted = false;
        this.onCancelCallback = onCancel;
        this.startTime = Date.now();
        this.lastTickTime = Date.now();
        this.totalSteps = totalSteps;
        this.currentStep = 1;
        this.transitionHistory = [];
        this.loopCount = 0;
        this.consecutiveRepeats = 0;

        if (this.container) this.container.style.display = 'block';
        if (this.badge) {
            this.badge.className = 'monitor-status-badge running';
        }
        if (this.stateText) this.stateText.innerHTML = 'ACTIVE &bull; COMPUTING';
        if (this.loopStatus) this.loopStatus.className = 'monitor-loop-status';
        if (this.loopText) this.loopText.innerText = 'Loop Guard: Normal (0 cycles)';
        if (this.stuckHint) {
            this.stuckHint.style.color = 'var(--accent-emerald)';
            this.stuckHint.innerText = 'System operating smoothly';
        }
        if (this.threadStatus) {
            this.threadStatus.style.color = 'var(--accent-emerald)';
            this.threadStatus.innerText = 'Responsive';
        }
        if (this.progressBar) {
            this.progressBar.style.width = '5%';
            this.progressBar.setAttribute('aria-valuenow', '5');
        }
        if (this.taskName) this.taskName.innerText = taskDescription;
        if (this.stepCounter) this.stepCounter.innerText = totalSteps > 1 ? `Step 1 of ${totalSteps}` : 'In progress...';

        // Start timer interval (updates every 100ms)
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = setInterval(() => this.onTimerTick(), 100);
    }

    onTimerTick() {
        if (this.aborted) return;
        const now = Date.now();
        const elapsed = (now - this.startTime) / 1000;
        const mins = Math.floor(elapsed / 60);
        const secs = Math.floor(elapsed % 60);
        const tenths = Math.floor((elapsed * 10) % 10);
        const timerStr = `⏱️ ${mins < 10 ? '0' + mins : mins}:${secs < 10 ? '0' + secs : secs}.${tenths}s`;

        if (this.timerText) this.timerText.innerText = timerStr;

        // Heartbeat stall check
        const gap = (now - this.lastTickTime) / 1000;
        if (this.heartbeatTick) {
            this.heartbeatTick.innerText = gap < 1 ? '< 1s ago' : `${gap.toFixed(1)}s ago`;
        }

        if (gap > 12) {
            // Potential stall warning
            if (this.badge) this.badge.className = 'monitor-status-badge stalled';
            if (this.stateText) this.stateText.innerText = '⚠️ HEAVY LOAD / POSSIBLE STALL';
            if (this.threadStatus) {
                this.threadStatus.style.color = 'var(--accent-rose)';
                this.threadStatus.innerText = 'High Latency';
            }
            if (this.stuckHint) {
                this.stuckHint.style.color = 'var(--accent-rose)';
                this.stuckHint.innerText = 'Calculations taking longer than expected. Force Stop takes effect at the end of the current epoch (neural network), tree (Random Forest) or 10 iterations (logistic/ridge); k-NN and Naive Bayes stop between models. A single tree or k-NN scan cannot be interrupted mid-computation.';
            }
        } else if (gap > 5) {
            // Elevated compute
            if (this.badge) this.badge.className = 'monitor-status-badge warning';
            if (this.stateText) this.stateText.innerText = '⚙️ BUSY COMPUTING';
            if (this.threadStatus) {
                this.threadStatus.style.color = 'var(--accent-amber)';
                this.threadStatus.innerText = 'Heavy Math';
            }
            if (this.stuckHint) {
                this.stuckHint.style.color = 'var(--accent-amber)';
                this.stuckHint.innerText = 'Complex gradient descent / tree construction in progress...';
            }
        } else {
            if (this.badge && this.badge.className.includes('warning')) {
                this.badge.className = 'monitor-status-badge running';
                if (this.stateText) this.stateText.innerHTML = 'ACTIVE &bull; COMPUTING';
                if (this.threadStatus) {
                    this.threadStatus.style.color = 'var(--accent-emerald)';
                    this.threadStatus.innerText = 'Responsive';
                }
                if (this.stuckHint) {
                    this.stuckHint.style.color = 'var(--accent-emerald)';
                    this.stuckHint.innerText = 'System operating smoothly';
                }
            }
        }
    }

    tick(stepIndex, taskDescription, details = '') {
        if (this.aborted) throw new Error("Operation halted by user.");
        this.lastTickTime = Date.now();
        this.currentStep = stepIndex;

        // 1. Repetition monitor (advisory only). Identical progress messages are normal during
        // multi-epoch training, so repetition is reported but never used to abort a run. Every
        // training loop in this app has a fixed iteration limit; genuine hangs are surfaced by the
        // heartbeat stall indicator instead.
        const transitionKey = `${stepIndex}:${taskDescription}`;
        const last = this.transitionHistory[this.transitionHistory.length - 1];
        this.consecutiveRepeats = (last && last.key === transitionKey) ? (this.consecutiveRepeats || 0) + 1 : 0;
        this.transitionHistory.push({ key: transitionKey, time: this.lastTickTime });
        if (this.transitionHistory.length > 200) this.transitionHistory.shift();

        if (this.consecutiveRepeats >= 50) {
            this.loopCount++;
            if (this.loopStatus) this.loopStatus.className = 'monitor-loop-status loop-detected';
            if (this.loopText) this.loopText.innerText = `⚠️ Progress message unchanged for ${this.consecutiveRepeats} updates (still running)`;
        } else {
            if (this.loopStatus) this.loopStatus.className = 'monitor-loop-status';
            if (this.loopText) this.loopText.innerText = 'Loop Guard: Normal (bounded iterations)';
        }

        // 2. Update UI
        if (this.taskName) this.taskName.innerText = taskDescription;
        if (this.stepCounter && this.totalSteps > 1) {
            this.stepCounter.innerText = `Step ${stepIndex} of ${this.totalSteps}`;
        }
        if (this.progressBar) {
            const pct = Math.min(100, Math.round((stepIndex / this.totalSteps) * 100));
            this.progressBar.style.width = `${pct}%`;
            this.progressBar.setAttribute('aria-valuenow', String(pct));
        }
    }

    finish(successMessage = 'Completed successfully') {
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = null;

        if (this.badge) this.badge.className = 'monitor-status-badge completed';
        if (this.stateText) this.stateText.innerText = '✓ COMPLETED';
        if (this.progressBar) {
            this.progressBar.style.width = '100%';
            this.progressBar.setAttribute('aria-valuenow', '100');
        }
        if (this.taskName) this.taskName.innerText = successMessage;
        if (this.stuckHint) {
            this.stuckHint.style.color = 'var(--accent-cyan)';
            this.stuckHint.innerText = 'Results rendered successfully';
        }

        setTimeout(() => {
            if (this.container && !this.aborted) {
                this.container.style.display = 'none';
            }
        }, 800);
    }

    forceAbort(reason = "Operation aborted by user.") {
        this.aborted = true;
        if (this.timerInterval) clearInterval(this.timerInterval);
        this.timerInterval = null;

        if (this.badge) this.badge.className = 'monitor-status-badge stalled';
        if (this.stateText) this.stateText.innerText = '🛑 ABORTED';
        if (this.taskName) this.taskName.innerText = reason;
        if (this.stuckHint) {
            this.stuckHint.style.color = 'var(--accent-rose)';
            this.stuckHint.innerText = 'Cancellation requested: remaining work is skipped at the next checkpoint.';
        }

        if (typeof this.onCancelCallback === 'function') {
            try {
                this.onCancelCallback(reason);
            } catch (e) {
                console.error("Cancel callback error:", e);
            }
        }

        setTimeout(() => {
            if (this.container) this.container.style.display = 'none';
        }, 2500);
    }
}

export { ExecutionWatchdog };
