import cors from 'cors'
import express from 'express'
import expressWS from 'express-ws'
import type * as http from 'http'

import type { Model, Task, TaskProvider } from '@epfml/discojs'

import { CONFIG } from './config.js'
import { Router } from './router/index.js'
import { TasksAndModels } from './tasks.js'

export class Disco {
  private readonly _app: express.Application
  private readonly tasksAndModels: TasksAndModels

  constructor () {
    this._app = express()
    this.tasksAndModels = new TasksAndModels()
  }

  public get server (): express.Application {
    return this._app
  }

  // Load tasks provided by default with disco server
  async addDefaultTasks (): Promise<void> {
    await this.tasksAndModels.loadDefaultTasks()
  }

  // If a model is not provided, its url must be provided in the task object
  async addTask (task: Task | TaskProvider, model?: Model | URL): Promise<void> {
    await this.tasksAndModels.addTaskAndModel(task, model)
  }

  serve (port?: number): http.Server {
    const wsApplier = expressWS(this.server, undefined, { leaveRouterUntouched: true })
    const app = wsApplier.app

    app.enable('trust proxy')
    app.use(cors())
    app.use(express.json({ limit: '50mb' }))
    app.use(express.urlencoded({ limit: '50mb', extended: false }))

    const baseRouter = new Router(wsApplier, this.tasksAndModels, CONFIG)
    app.use('/', baseRouter.router)

    const server = app.listen(port ?? CONFIG.serverPort, () => {
      console.log(`Disco Server listening on ${CONFIG.serverUrl.href}`)
    })

    console.info('Disco Server initially loaded the tasks below\n')
    console.table(
      Array.from(this.tasksAndModels.tasksAndModels).map(([task]) => {
        return {
          ID: task.id,
          Title: task.displayInformation.taskTitle,
          'Data Type': task.trainingInformation.dataType,
          Scheme: task.trainingInformation.scheme
        }
      })
    )
    console.log()

    return server
  }
}

export async function runDefaultServer (port?: number): Promise<http.Server> {
  const disco = new Disco()
  await disco.addDefaultTasks()
  return disco.serve(port)
}
