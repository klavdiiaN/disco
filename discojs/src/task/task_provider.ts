import type { Model, Task } from '../index.js'

export interface TaskProvider {
  getTask: (numClasses?: number) => Task
  // Create the corresponding model ready for training (compiled)
  getModel: (numClasses?: number) => Promise<Model>
}
