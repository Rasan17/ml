// multiple-imputation.js: multiple imputation (MICE) for the prediction models in the standard workflow.
// The imputation models are fitted on the training patients only (without the outcome); one model is
// trained per completed dataset and the predictions are pooled (ModelEngine.createMIPooled). Testing and new
// patients are imputed with the same fitted chains by DataEngine.transformSingleRow.
/* global DataEngine, ModelEngine */
import { fitModel } from './compute.js';

// Number of imputations chosen in Step 4 (0 = single imputation)
function readMultipleImputation() {
    const v = document.getElementById('mi-select')?.value || 'single';
    return v === 'mice10' ? 10 : v === 'mice5' ? 5 : 0;
}

/**
 * Multiply imputes raw training rows and returns the encoding fitted on all completed copies (prep, whose
 * X / y are the first copy, for counts and displays) and the encoded copies (sets) to train on.
 */
function prepareMultipleImputation(rawRows, columnConfigs, m, seed) {
    const mi = DataEngine.miceFit(rawRows, columnConfigs, { m, seed });
    const stacked = [].concat(...mi.datasets);
    const prep = DataEngine.prepareFeaturesAndTarget(stacked, columnConfigs);
    prep.imputation = { configs: JSON.parse(JSON.stringify(columnConfigs)), mice: mi.imputer };
    const sets = mi.datasets.map(d => {
        const t = DataEngine.prepareTestingDataset(d, prep);
        return { X: t.X, y: t.y };
    });
    prep.X = sets[0].X;
    prep.y = sets[0].y;
    prep.multipleImputation = { m, imputedColumns: mi.imputedColumns, rows: rawRows.length };
    return { prep, sets, imputedColumns: mi.imputedColumns };
}

/** Trains one model per completed copy and returns the pooled model */
async function fitPooledModel(modelId, isClassification, numClasses, hp, sets, opts = {}) {
    const models = [];
    for (let l = 0; l < sets.length; l++) {
        if (opts.onImputation) opts.onImputation(l + 1, sets.length);
        models.push(await fitModel(modelId, isClassification, numClasses, hp, sets[l].X, sets[l].y, opts));
    }
    return ModelEngine.createMIPooled(models);
}

export { readMultipleImputation, prepareMultipleImputation, fitPooledModel };
