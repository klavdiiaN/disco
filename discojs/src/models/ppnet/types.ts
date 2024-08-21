import type tf from '@tensorflow/tfjs'
import * as fs from 'fs';

export interface TrainingCallbacks {
  onEpochEnd?: (epoch: number, logs?: tf.Logs) => Promise<void> 
}

// extension which allows to write logs into a JSON file after each epoch
export class saveResults implements TrainingCallbacks {
  private filePath: string;

  constructor(filePath: string) {
      this.filePath = filePath;
  }

  async onEpochEnd(epoch: number, logs?: tf.Logs): Promise<void> {
      if (logs) {
          const data = {
              epoch: epoch,
              loss: logs.val_loss,
              accuracy: logs.balanced_acc,
              sensitivity: logs.sensitivity,
              specificity: logs.specificity
          };
          let dataArray = [];

          // Read existing data if the file exists
          if (fs.existsSync(this.filePath)) {
            const existingData = fs.readFileSync(this.filePath, 'utf8');
            dataArray = JSON.parse(existingData);
        }

        // Append the new entry to the array
        dataArray.push(data);

        // Write the updated array back to the file
        fs.writeFileSync(this.filePath, JSON.stringify(dataArray, null, 2));
        console.log(`Metrics saved for epoch ${epoch}`);
    }
  }
}