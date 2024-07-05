"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs");
var config_js_1 = require("./config.js");
var utils_js_1 = require("./utils.js");
function resolveConfig(config) {
    return __assign(__assign({}, config_js_1.DEFAULT_CONFIG), config);
}
;
;
var protoClassId = (0, utils_js_1.getProtoClassIdx)({
    prototypeShape: [200, 1, 1, 128],
    numClasses: 20
});
var ppLoss = (0, utils_js_1.protoPartLoss)({
    prototypeShape: [200, 1, 1, 128]
}, protoClassId);
// Custom balanced accuracy metric function
function balancedAccuracy(yTrue, yPred) {
    // Calculate true positives, true negatives, false positives, and false negatives
    var truePositives = yTrue.mul(yPred).sum().cast('float32');
    var trueNegatives = yTrue.equal(0).logicalAnd(yPred.equal(0)).sum().cast('float32');
    var falsePositives = yTrue.equal(0).logicalAnd(yPred.equal(1)).sum().cast('float32');
    var falseNegatives = yTrue.equal(1).logicalAnd(yPred.equal(0)).sum().cast('float32');
    // Calculate sensitivity and specificity
    var sensitivity = truePositives.div(truePositives.add(falseNegatives)).cast('float32');
    var specificity = trueNegatives.div(trueNegatives.add(falsePositives)).cast('float32');
    // Calculate balanced accuracy
    var balancedAcc = sensitivity.add(specificity).div(tf.scalar(2));
    return [balancedAcc.arraySync(), sensitivity.arraySync(), specificity.arraySync()];
}
;
function evaluate(model, dataset, cfg) {
    return __awaiter(this, void 0, void 0, function () {
        var config, totalLoss, totalBalancedAcc, totalBatches, totalSens, totalSpec, avgLoss, avgBalancedAcc, avgSens, avgSpec;
        var _this = this;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    config = resolveConfig(cfg);
                    totalLoss = 0;
                    totalBalancedAcc = 0;
                    totalBatches = 0;
                    totalSens = 0;
                    totalSpec = 0;
                    return [4 /*yield*/, dataset.batch(config.batchSizeEval).forEachAsync(function (batch) { return __awaiter(_this, void 0, void 0, function () {
                            var _a, xs, ys, output, lossTensor, loss, predictions, _b, balancedAcc, sensitivity, specificity;
                            return __generator(this, function (_c) {
                                switch (_c.label) {
                                    case 0:
                                        _a = batch, xs = _a[0], ys = _a[1];
                                        output = model.predict(xs);
                                        lossTensor = ppLoss(ys, output[0]);
                                        return [4 /*yield*/, lossTensor.data()];
                                    case 1:
                                        loss = _c.sent();
                                        totalLoss += loss[0];
                                        predictions = output[0].slice(0, config.numClasses);
                                        _b = balancedAccuracy(ys, predictions), balancedAcc = _b[0], sensitivity = _b[1], specificity = _b[2];
                                        totalBalancedAcc += balancedAcc.reduce(function (acc, val) { return acc + val; }, 0);
                                        totalSens += sensitivity.reduce(function (acc, val) { return acc + val; }, 0);
                                        totalSpec += specificity.reduce(function (acc, val) { return acc + val; }, 0);
                                        totalBatches++;
                                        return [2 /*return*/];
                                }
                            });
                        }); })];
                case 1:
                    _a.sent();
                    avgLoss = totalLoss / totalBatches;
                    avgBalancedAcc = totalBalancedAcc / totalBatches;
                    avgSens = totalSens / totalBatches;
                    avgSpec = totalSpec / totalBatches;
                    return [2 /*return*/, { val_loss: avgLoss,
                            balanced_acc: avgBalancedAcc,
                            sensitivity: avgSens,
                            specificity: avgSpec }];
            }
        });
    });
}
exports.default = evaluate;
;
