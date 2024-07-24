import * as tf from '@tensorflow/tfjs';

// This function retrieves the class index to which a prototype belongs
export function getProtoClassIdx (cfg: any): tf.Tensor {
    const config = Object.assign({}, cfg);
    const numClasses = config.numClasses;
    const numPrototypes = config.prototypeShape[0];

    const numPrototypePerClasses = Math.floor(numPrototypes / numClasses);
    const protoClassId = tf.zeros([numPrototypes, numClasses]);
    let protoClassIdBuffer = tf.buffer(protoClassId.shape, protoClassId.dtype, protoClassId.dataSync())

    for (let j = 0; j < numPrototypes; j++) {
        protoClassIdBuffer.set(1, j, Math.floor(j / numPrototypePerClasses));
    }

    return protoClassIdBuffer.toTensor();
};

// This function implements a custom loss function for ProtoPNet which consists of three terms: cross entropy, cluster, and separation cost
export function protoPartLoss (cfg: any, protoClassId: tf.Tensor) {
    return (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor => {
        return tf.tidy(() => {
        const predictions = yPred.slice([0, 0], [-1, cfg.numClasses]); // since output tensor is a combination of two, we first retrieve only the class logits
        //console.log('Predictions:', predictions.shape);
        const minDistances = yPred.slice([0, cfg.numClasses], [-1, cfg.prototypeShape[0]]); // retrieve the min distances too
        //console.log('OneHot Labels:', oneHotLabels.shape);
        const crossEntropy = tf.losses.softmaxCrossEntropy(yTrue, predictions);
        //console.log('CrossEntropy:', crossEntropy);
        const labels = yTrue.argMax(1);
        //console.log('Labels:', labels);

        // cluster cost ensures that each training image has at least one patch which is close to a prototype of its own class 
        const maxDistance = cfg.prototypeShape[1] * cfg.prototypeShape[2] * cfg.prototypeShape[3];
        const prototypesOfCorrectClass = tf.transpose(protoClassId.gather(labels, 1));

        const invertedDistances = tf.max(
            tf.mul(tf.sub(maxDistance, minDistances), prototypesOfCorrectClass),
            1
        );

        const clusterCost = tf.mean(tf.sub(maxDistance, invertedDistances)); 
        //console.log('Cluster Cost:', clusterCost);

        // separation cost forces every latent patch of a training image to stay away from the prototypes not of its own class
        const prototypesOfWrongClass = tf.sub(tf.scalar(1), prototypesOfCorrectClass);
        const invertedDistancesNontarget = tf.max(
            tf.mul(tf.sub(maxDistance, minDistances), prototypesOfWrongClass),
            1
        );

        const separationCost = tf.mean(tf.sub(maxDistance, invertedDistancesNontarget));
        //console.log('Separation cost:', separationCost);
        const finalLoss = tf.addN([crossEntropy,
            tf.mul(tf.scalar(0.8), clusterCost),
            tf.mul(tf.scalar(-0.08), separationCost)
        ]); // combine all terms into the final loss function (coefficients are taken from the original ProtoPNet paper)

        labels.dispose(); // Dispose tensors that are not needed after their use
            prototypesOfCorrectClass.dispose();
            prototypesOfWrongClass.dispose();
            invertedDistances.dispose();
            invertedDistancesNontarget.dispose();

            return finalLoss;
        });
    }
};

