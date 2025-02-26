import type * as http from 'http'

import type { Task } from '@epfml/discojs'
import { aggregator as aggregators, client as clients, defaultTasks } from '@epfml/discojs'

import { startServer } from '../../src/index.js'

const TASK = defaultTasks.titanic.getTask()

function test (
  name: string,
  Client: new (url: URL, task: Task, aggregator: aggregators.Aggregator) => clients.Client,
  Aggregator: new () => aggregators.Aggregator
): void {
  describe(`decentralized ${name} client`, function () {
    this.timeout(30_000)

    let server: http.Server
    let url: URL
    beforeEach(async () => { [server, url] = await startServer() })
    afterEach(() => { server?.close() })

    it('connect and disconnect from valid task', async () => {
      const aggregator = new Aggregator()
      const client = new Client(url, TASK, aggregator)

      await client.connect()
      await client.disconnect()
    })

    // TODO @s314cy: update
    // it('connect to other nodes', async () => {
    //   const users = List(await Promise.all([
    //     getClient(Client, server, TASK, new Aggregator(TASK)),
    //     getClient(Client, server, TASK, new Aggregator(TASK)),
    //     getClient(Client, server, TASK, new Aggregator(TASK))
    //   ]))
    //   try {
    //     await Promise.all(users.map(async (u) => await u.connect()))

    //     const wss = List.of(
    //       WeightsContainer.of(tf.tensor(0)),
    //       WeightsContainer.of(tf.tensor(1)),
    //       WeightsContainer.of(tf.tensor(2))
    //     )

    //     const tis = users.map(() => new informant.DecentralizedInformant(TASK, 0))

    //     // wait for others to connect
    //     await new Promise((resolve) => setTimeout(resolve, 1_000))

    //     await Promise.all(
    //       users.zip(wss, tis)
    //         .map(async ([u, ws, ti]) => await u.onRoundBeginCommunication(ws, 0, ti))
    //         .toArray()
    //     )

    //     tis.forEach((ti) => expect(users.size).to.eq(ti.participants()))
    //   } finally {
    //     await Promise.all(users.map(async (u) => await u.disconnect()))
    //   }
    // })
  })
}

test('cleartext', clients.decentralized.DecentralizedClient, aggregators.MeanAggregator)
test('secure', clients.decentralized.DecentralizedClient, aggregators.SecureAggregator)
