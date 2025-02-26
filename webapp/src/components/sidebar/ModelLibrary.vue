<template>
  <TippyContainer>
    <template #title> Model Library </template>
    <template #icon>
      <StackIcon />
    </template>
    <template #content>
      <!-- Model list -->
      <TippyCard>
        <template #title> My Model Library </template>
        <template #content>
          <span class="text-s">
            <p v-if="memoryStore.useIndexedDB">
              List of trained models that were saved. You can download
              pre-trained models in the
              <button class="text-blue-400" @click="switchToEvaluate()">
                Evaluation page</button
              >.
            </p>
            <p v-else>
              The model library is currently unavailable. You can turn it on in
              the settings below.
            </p>
          </span>
          <div v-if="memoryStore.useIndexedDB" class="space-y-4">
            <button
              v-for="[path, metadata] in memoryStore.models"
              :key="path"
              class="flex items-center justify-between px-4 py-2 space-x-4 outline outline-1 outline-slate-300 rounded-md transition-colors duration-200 text-slate-600 hover:text-slate-800 hover:outline-slate-800 focus:outline-none focus:ring-1 focus:ring-slate-800"
            >
              <div class="cursor-pointer w-2/3" @click="openTesting(path)">
                <span>
                  {{ metadata.name.slice(0, 16) }}
                  <span
                    v-if="
                      metadata.version !== undefined && metadata.version !== 0
                    "
                  >
                    ({{ metadata.version }}) </span
                  ><br />
                  <span class="text-xs">
                    {{ metadata.date }} at {{ metadata.hours }} <br />
                    {{ metadata.fileSize }} kB
                  </span>
                </span>
              </div>
              <div class="w-1/9">
                <ModelButton
                  event="delete-model"
                  hover="Delete"
                  @delete-model="deleteModel(path)"
                >
                  <Bin2Icon />
                </ModelButton>
              </div>
              <div class="w-1/9">
                <ModelButton
                  event="download-model"
                  hover="Download"
                  @download-model="downloadModel(path)"
                >
                  <Download2Icon />
                </ModelButton>
              </div>
              <div class="w-1/9">
                <ModelButton
                  event="load-model"
                  hover="Load for next training"
                  @load-model="loadModel(path)"
                >
                  <LoadIcon />
                </ModelButton>
              </div>
            </button>
          </div>
        </template>
      </TippyCard>
      <div class="overflow-hidden hover:overflow-y-auto">
        <!-- IndexedDB -->
        <TippyCard>
          <template #title> Settings </template>
          <template #content>
            <span class="text-s">
              Turn on to get storage options for your trained models. This uses
              your browser's own database, namely
              <button class="text-blue-400">
                <a
                  href="https://en.wikipedia.org/wiki/Indexed_Database_API"
                  target="_blank"
                >
                  IndexedDB
                </a>
              </button>
              .
              <br />
            </span>

            <div class="flex items-center justify-center">
              <button
                class="flex items-center justify-center px-4 py-2 space-x-4 outline outline-1 outline-slate-300 rounded-md transition-colors duration-200 text-slate-600 hover:text-slate-800 hover:outline-slate-800 focus:outline-none focus:ring-1 focus:ring-slate-800"
                @click="toggleIndexedDB()"
              >
                <span class="text-s"> Use model library </span>
                <div class="relative focus:outline-none">
                  <div
                    class="w-12 h-6 transition rounded-full outline-none bg-slate-200"
                  />
                  <div
                    class="absolute top-0 left-0 inline-flex w-6 h-6 transition-all duration-200 ease-in-out transform scale-110 rounded-full shadow-sm"
                    :class="{
                      'translate-x-0 bg-slate-300': !memoryStore.useIndexedDB,
                      'translate-x-6 bg-disco-blue': memoryStore.useIndexedDB,
                    }"
                  />
                </div>
              </button>
            </div>
          </template>
        </TippyCard>
      </div>
    </template>
  </TippyContainer>
</template>
<script lang="ts" setup>
import { List } from "immutable";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";

import type { Path } from "@epfml/discojs";
import { EmptyMemory } from "@epfml/discojs";
import { IndexedDB } from "@epfml/discojs-web";

import { useToaster } from "@/composables/toaster";
import type { ModelMetadata } from "@/store/memory";
import { useMemoryStore } from "@/store/memory";
import { useValidationStore } from "@/store/validation";
import ModelButton from "./simple/ModelButton.vue";
import Bin2Icon from "@/assets/svg/Bin2Icon.vue";
import Download2Icon from "@/assets/svg/Download2Icon.vue";
import LoadIcon from "@/assets/svg/LoadIcon.vue";
import StackIcon from "@/assets/svg/StackIcon.vue";
import TippyCard from "@/components/sidebar/containers/TippyCard.vue";
import TippyContainer from "@/components/sidebar/containers/TippyContainer.vue";

const emit = defineEmits<{
  "close-panel": [];
}>();

const toaster = useToaster();
const router = useRouter();
const memoryStore = useMemoryStore();
const validationStore = useValidationStore();

const models = ref(List<[string, ModelMetadata]>());
memoryStore.$subscribe((_, state) => {
  models.value = List(state.models.sort());
});

const memory = memoryStore.useIndexedDB ? new IndexedDB() : new EmptyMemory();

onMounted(async () => {
  await memoryStore.initModels();
});

function toggleIndexedDB() {
  memoryStore.setIndexedDB(
    !memoryStore.useIndexedDB && Boolean(window.indexedDB),
  );
}

function switchToEvaluate(): void {
  router.push({ path: "/evaluate" });
  emit("close-panel");
}
function openTesting(path: Path) {
  validationStore.setModel(path);
  router.push({ path: "/evaluate" });
}

 async function loadModel(path: Path) {
  const modelInfo = memory.getModelInfo(path)
  if (modelInfo === undefined) {
    throw new Error('not such model')
  }
   if (modelInfo.type !== 'working') {
    try {
      await memory.loadModel(path)
      await memoryStore.initModels()
    } catch (e) {
      let msg = 'unable to load model'
      if (e instanceof Error) {
        msg += `: ${e.message}`
      }
      toaster.error(msg)
    }
    toaster.success(`Loaded ${modelInfo.name}, ready for next training session.`)
  } else {
    toaster.error('Model is already loaded')
  }
}

async function downloadModel(path: Path) {
  await memory.downloadModel(path);
}

async function deleteModel(path: Path): Promise<void> {
  try {
    await memoryStore.deleteModel(path);
    await memory.deleteModel(path);
    toaster.success("Successfully deleted the model");
  } catch (e) {
    let msg = "unable to delete model";
    if (e instanceof Error) msg += `: ${e.message}`;
    toaster.error(msg);
  }
}
</script>
