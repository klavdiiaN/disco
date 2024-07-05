import { Task, data, TaskProvider, Model, models } from '../index.js'

export const ppnet: TaskProvider = {
  getTask (numClasses?: number): Task {
    return {
      id: 'ppnet',
      displayInformation: {
        taskTitle: 'Prototypical Part Network',
        summary: {
          preview: 'Data bias identification using prototypical part neural network',
          overview: 'Interpreting prototypical parts to leverage explainability in deep learning models in decentralized setting and identifying data bias among clients.'
        },
        //limitations: 'This is the simplified version of the fully functional Prototypical Part Network in Python',
        //tradeoffs: 'Model simplification vs. performance',
        dataFormatInformation: 'CheXpert, benchmark dataset of human chest X-rays' //'A subset of birds classification dataset',
      },
      trainingInformation: {
        modelID: 'ppnet',
        epochs: 2, // NUMBER OF COMMUNICATION ROUNDS
        roundDuration: 10, // NUMBER OF LOCAL TRAINING EPOCHS
        validationSplit: 0.1,
        batchSize: 80,
        preprocessingFunctions: [data.ImagePreprocessing.Normalize, data.ImagePreprocessing.Resize],
        dataType: 'image',
        IMAGE_H: 224,
        IMAGE_W: 224,
        LABEL_LIST: [],
          //'negative', 'positive'
            /*'Black Footed Albatross', 
            'Laysan Albatross', 
            'Sooty Albatross', 
            'Groove Billed Ani', 
            'Crested Auklet', 
            'Least Auklet',
            'Parakeet Auklet',
            'Rhinocerus Auklet',
            'Brewe Blackbird',
            'Red Wing Blackbird',
            'Rusty Blackbird',
            'Yellow Headed Blackbird',
            'Bobolink',
            'Indigo Bunting',
            'Lazuli Bunting',
            'Painted Bunting',
            'Cardinal',
            'Spotted Catbird',
            'Gray Catbird',
            'Yellow Breasted Chat'*/
        numClasses: numClasses,
        scheme: 'federated', // secure aggregation not yet implemented for federated
        noiseScale: undefined,
        clippingRadius: undefined,
        tensorBackend: 'tfjs'
      }
    }
  },

  getModel (numClasses?: number): Promise<Model> {
    return models.PPNet.createInstance(numClasses)
  }
}