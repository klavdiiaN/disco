<template>
  <div class="grid grid-cols-1">
    <IconCard class="justify-self-center w-full">
      <template #title> My dataset </template>
      <template #icon>
        <Upload />
      </template>
      <template #content>
        <FileSelection @files="setSingleFile($event)" />
      </template>
    </IconCard>
  </div>
</template>

<script lang="ts" setup>
import { Set } from "immutable";

import type { Dataset, Text } from "@epfml/discojs";
import { loadText } from "@epfml/discojs-web";

import Upload from "@/assets/svg/Upload.vue";
import IconCard from "@/components/containers/IconCard.vue";
import FileSelection from "./FileSelection.vue";

const emit = defineEmits<{
  dataset: [dataset: Dataset<Text> | undefined];
}>();

async function setSingleFile(files: Set<File> | undefined): Promise<void> {
  if (files === undefined) {
    emit("dataset", undefined);
    return;
  }

  const file = files.first();
  if (file === undefined || files.size > 1)
    throw new Error("excepted a single file");

  const dataset = loadText(file)

  emit("dataset", dataset);
}
</script>
