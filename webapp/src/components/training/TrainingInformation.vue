<template>
  <div class="space-y-4 md:space-y-8">
    <!-- Fancy training statistics -->
    <div class="flex flex-wrap justify-center gap-4 md:gap-8">
      <IconCardSmall
        header="current round"
        :text="`${roundsCount}`"
        class="w-72 shrink-0"
      >
        <Timer />
      </IconCardSmall>
      <IconCardSmall
        header="current epoch"
        :text="`${epochsCount}`"
        class="w-72 shrink-0"
      >
        <Timer />
      </IconCardSmall>
      <IconCardSmall
        header="current batch"
        :text="`${batchesCount}`"
        class="w-72 shrink-0"
      >
        <Timer />
      </IconCardSmall>

      <IconCardSmall
        header="current # of participants"
        :text="`${participants.current}`"
        class="w-72 shrink-0"
      >
        <People />
      </IconCardSmall>
      <IconCardSmall
        header="average # of participants"
        :text="`${participants.average}`"
        class="w-72 shrink-0"
      >
        <People />
      </IconCardSmall>
    </div>

    <!-- Training and validation loss charts -->
    <div
      class="flex flex-col md:grid gap-4 md:gap-8"
      :class="hasValidationData ? 'md:grid-cols-2' : ''"
    >
      <!-- Training loss users chart -->
      <IconCard>
        <!-- Card header -->
        <template #title> Training Loss of the Model </template>
        <template #content>
          <span class="text-2xl font-medium text-slate-500">
            {{ (lastEpoch?.training.loss ?? 0).toFixed(2) }}
          </span>
          <span class="text-sm font-medium text-slate-500">
            training loss
          </span>
          <!-- Chart -->
          <ApexChart
            width="100%"
            height="200"
            type="area"
            :options="lossChartsOptions"
            :series="[{ name: 'Training loss', data: lossSeries.training }]"
          />
        </template>
      </IconCard>

      <!-- Validation Loss users chart -->
      <IconCard v-if="hasValidationData">
        <!-- Card header -->
        <template #title> Validation Loss of the Model </template>
        <template #content>
          <span class="text-2xl font-medium text-slate-500">
            {{ (lastEpoch?.validation?.loss ?? 0).toFixed(2) }}
          </span>
          <span class="text-sm font-medium text-slate-500">
            validation loss
          </span>
          <!-- Chart -->
          <ApexChart
            width="100%"
            height="200"
            type="area"
            :options="lossChartsOptions"
            :series="[{ name: 'Validation loss', data: lossSeries.validation }]"
          />
        </template>
      </IconCard>
    </div>
    <!-- Training and validation accuracy charts -->
    <div
      class="flex flex-col md:grid gap-4 md:gap-8"
      :class="hasValidationData ? 'md:grid-cols-2' : ''"
    >
      <!-- Training Accuracy users chart -->
      <IconCard>
        <!-- Card header -->
        <template #title> Training Accuracy of the Model </template>
        <template #content>
          <span class="text-2xl font-medium text-slate-500">
            {{ percent(lastEpoch?.training.accuracy ?? 0) }}
          </span>
          <span class="text-sm font-medium text-slate-500">
            % of training accuracy
          </span>
          <!-- Chart -->
          <ApexChart
            width="100%"
            height="200"
            type="area"
            :options="accuracyChartsOptions"
            :series="[
              { name: 'Training accuracy', data: accuracySeries.training },
            ]"
          />
        </template>
      </IconCard>

      <!-- Validation Accuracy users chart -->
      <IconCard v-if="hasValidationData">
        <!-- Card header -->
        <template #title> Validation Accuracy of the Model </template>
        <template #content>
          <span class="text-2xl font-medium text-slate-500">
            {{ percent(lastEpoch?.validation?.accuracy ?? 0) }}
          </span>
          <span class="text-sm font-medium text-slate-500">
            % of validation accuracy
          </span>
          <!-- Chart -->
          <ApexChart
            width="100%"
            height="200"
            type="area"
            :options="accuracyChartsOptions"
            :series="[
              { name: 'Validation accuracy', data: accuracySeries.validation },
            ]"
          />
        </template>
      </IconCard>
    </div>

    <!-- Training logs -->
    <IconCard>
      <template #title> Training Logs </template>
      <template #icon>
        <Contact />
      </template>
      <template #content>
        <!-- Scrollable training logs -->
        <div id="mapHeader" class="max-h-80 overflow-y-auto">
          <ul class="grid grid-cols-1">
            <li
              v-for="(message, index) in props.messages"
              :key="index"
              class="border-slate-400"
            >
              <span
                style="white-space: pre-line"
                class="text-sm text-slate-500"
              >
                {{ message }}
              </span>
            </li>
          </ul>
        </div>
      </template>
    </IconCard>
  </div>
</template>

<script setup lang="ts">
import { List } from "immutable";
import { computed } from "vue";
import ApexChart from "vue3-apexcharts";

import type { BatchLogs, EpochLogs, RoundLogs } from "@epfml/discojs";

import IconCardSmall from "@/components/containers/IconCardSmall.vue";
import IconCard from "@/components/containers/IconCard.vue";
import Timer from "@/assets/svg/Timer.vue";
import People from "@/assets/svg/People.vue";
import Contact from "@/assets/svg/Contact.vue";

const props = defineProps<{
  rounds: List<RoundLogs & { participants: number }>;
  epochsOfRound: List<EpochLogs>;
  batchesOfEpoch: List<BatchLogs>;
  hasValidationData: boolean; // TODO infer from logs
  messages: List<string>; // TODO why do we want messages?
}>();

const participants = computed(() => ({
  current: props.rounds.last()?.participants ?? 0,
  average:
    props.rounds.size > 0
      ? props.rounds.reduce((acc, round) => acc + round.participants, 0) /
        props.rounds.size
      : 0,
}));

const batchesCount = computed(() => props.batchesOfEpoch.size);
const epochsCount = computed(() => props.epochsOfRound.size);
const roundsCount = computed(() => props.rounds.size);

const allEpochs = computed(() =>
  props.rounds.flatMap((round) => round.epochs).concat(props.epochsOfRound),
);
const lastEpoch = computed(() => allEpochs.value.last());

const accuracySeries = computed(() =>
  allEpochs.value
    .map((epoch) => ({
      training: epoch.training.accuracy * 100,
      validation: (epoch.validation?.accuracy ?? 0) * 100,
    }))
    .reduce(
      ({ training, validation }, cur) => ({
        training: training.concat([cur.training]),
        validation: validation.concat([cur.validation]),
      }),
      {
        training: [] as number[],
        validation: [] as number[],
      },
    ),
);
const lossSeries = computed(() =>
  allEpochs.value
    .map((epoch) => ({
      training: epoch.training.loss,
      validation: epoch.validation?.loss ?? 0,
    }))
    .reduce(
      ({ training, validation }, cur) => ({
        training: training.concat([cur.training]),
        validation: validation.concat([cur.validation]),
      }),
      {
        training: [] as number[],
        validation: [] as number[],
      },
    ),
);

const commonChartsOptions = {
  chart: {
    animations: {
      enabled: true,
      easing: "linear",
      dynamicAnimation: { speed: 1000 },
    },
    toolbar: { show: false },
    zoom: { enabled: false },
  },
  dataLabels: { enabled: false },
  colors: ["#6096BA"],
  fill: {
    colors: ["#E2E8F0"],
    type: "solid",
    opacity: 0.6,
  },
  stroke: { curve: "smooth" },
  markers: { size: 0.5 },
  grid: {
    xaxis: { lines: { show: false } },
    yaxis: { lines: { show: false } },
  },
  xaxis: { labels: { show: false } },
  legend: { show: false },
  tooltip: { enabled: true },
};

const accuracyChartsOptions = {
  ...commonChartsOptions,
  yaxis: {
    max: 100,
    min: 0,
    labels: {
      show: true,
      formatter: (value: number) => value.toFixed(0),
    },
  },
};

const lossChartsOptions = computed(() => {
  const maxVal = Math.max(
    lossSeries.value.training.reduce((max, e) => Math.max(max, e), 0),
    lossSeries.value.validation.reduce((max, e) => Math.max(max, e), 0),
  );
  // if Math.max returns -inf or 0, set the max to 10 arbitrarily
  const yAxisMax = maxVal > 0 ? maxVal : 10;

  return {
    ...commonChartsOptions,
    yaxis: {
      max: yAxisMax,
      min: 0,
      labels: {
        show: true,
        formatter: (n: number) => n.toFixed(2),
      },
    },
  };
});

function percent(n: number): string {
  return (n * 100).toFixed(2);
}
</script>
