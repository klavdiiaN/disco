<template>
  <div class="space-y-8 mt-8 md:mt-16">
    <div class="flex flex-col gap-4 mt-8">
      <!-- In case no tasks were retrieved, suggest reloading the page -->
      <ButtonsCard
        v-if="tasks.size === 0"
        :buttons="List.of(['reload page', () => router.go(0)])"
        class="mx-auto"
      >
        <template #title>
          The server is unreachable
        </template>
          Please reload the app and make sure you are connected to internet. If the error persists please <a class='underline text-blue-400' target="_blank" href='https://join.slack.com/t/disco-decentralized/shared_invite/zt-fpsb7c9h-1M9hnbaSonZ7lAgJRTyNsw'>reach out on Slack</a>.
      </ButtonsCard>

      <!-- Tasks could be retrieved, display them alphabetically -->
      <div
        id="tasks"
        class="contents"
        v-else
      >
        <IconCard class="justify-self-center w-full">
        <template #title>
          What are <DISCOllaboratives />?
        </template>
          <template #icon>
            <Tasks/>
          </template>
          <template #content>
            <DISCOllaboratives /> are machine learning tasks, such as diagnosing COVID from ultrasounds or classifying hand written digits, that users can join to train and contribute to with their own data. To give you a sense of <DISCO />, we pre-defined some tasks
          along with some example datasets. The end goal is for users to create their own custom <DISCOllaborative /> and collaboratively train machine learning models.
          <br>By participating to a task, you can either choose to train a model with your own data only or join a collaborative training session with other users.
          If you want to bring your own collaborative task into <DISCO />, you can do so by <button
            class="text-blue-400"
            @click="goToCreateTask()"
          >creating a new <DISCOllaborative /></button>.
          <br/><br/> <b>The data you connect is only used locally and is never uploaded or shared with anyone. Data always stays on your device.</b>
          </template>
        </IconCard>
        <div
          v-for="task in sortedTasks"
          :id="task.id"
          :key="task.id"
        >
          <ButtonsCard
            buttons-justify="start"
            :buttons="List.of(['participate', () => toTask(task)])"
          >
            <template #title>
              <div class="flex flex-row justify-between flex-wrap">
                <div>{{ task.displayInformation.taskTitle }}</div>
                <div class="flex flex-row shrink-0 justify-end gap-1">
                  <div class="px-2 py-1 rounded-md flex items-center" :class="getSchemeColor(task)">
                    <div class="text-xs font-semibold text-slate-500 ">{{ task.trainingInformation.scheme.toUpperCase() }}</div>
                  </div>
                  <div class="px-2 py-1 rounded-md flex items-center" :class="getDataTypeColor(task)">
                    <div class="text-xs font-semibold text-slate-500">{{ task.trainingInformation.dataType.toUpperCase() }}</div>
                  </div>
                </div>
              </div>
            </template>

            <div v-html="task.displayInformation.summary.preview" />
          </ButtonsCard>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'

import { List } from "immutable";

import type { Task } from '@epfml/discojs'

import { useTasksStore } from '@/store/tasks'
import { useTrainingStore } from '@/store/training'
import ButtonsCard from '@/components/containers/ButtonsCard.vue'
import IconCard from '@/components/containers/IconCard.vue'
import DISCO from '@/components/simple/DISCO.vue'
import Tasks from '@/assets/svg/Tasks.vue'
import DISCOllaborative from '@/components/simple/DISCOllaborative.vue'
import DISCOllaboratives from '@/components/simple/DISCOllaboratives.vue'

const router = useRouter()
const trainingStore = useTrainingStore()
const { tasks } = storeToRefs(useTasksStore())

const sortedTasks = computed(() => [...tasks.value.values()].sort(
  (task1, task2) => task1.displayInformation.taskTitle.localeCompare(task2.displayInformation.taskTitle)
))

function getSchemeColor(task: Task): string {
  switch (task.trainingInformation.scheme) {
    case 'decentralized':
      return 'bg-orange-200'
    case 'federated':
      return 'bg-purple-200'
    case 'local':
      return 'bg-blue-200'
  }
}
function getDataTypeColor(task: Task): string {
  switch (task.trainingInformation.dataType) {
    case 'image':
      return 'bg-yellow-200'
    case 'tabular':
      return 'bg-blue-200'
    case 'text':
      return 'bg-green-200'
  }
}

const toTask = (task: Task): void => {
  trainingStore.setTask(task.id)
  router.push(`/${task.id}`)
}

const goToCreateTask = (): void => {
  router.push({ path: '/create' })
}
</script>
