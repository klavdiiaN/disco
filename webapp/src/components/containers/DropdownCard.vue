<script setup lang="ts">

import { ref } from 'vue'

import IconCard from '@/components/containers/IconCard.vue'
import UpArrow from '@/assets/svg/UpArrow.vue'
import DownArrow from '@/assets/svg/DownArrow.vue'

const displayed = ref(true)
const toggle = () => { displayed.value = !displayed.value }

</script>

<template>
  <IconCard :with-content="displayed">
    <template #title>
        <button
          class="focus:outline-none"
          @click="toggle"
        >
          <slot name="title" />
        </button>
    </template>
    <template #icon>
      <div class="flex">
        <button
          class="focus:outline-none"
          @click="toggle"
        >
          <UpArrow
            v-show="displayed"
          />
          <DownArrow
            v-show="!displayed"
          />
        </button>
      </div>
    </template>
    <template #content>
      <slot name="content" />
    </template>
  </IconCard>
</template>
