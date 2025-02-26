<template>
  <FieldArray
    :id="props.field.id"
    v-slot="{ fields, push, remove }"
    :name="props.field.id"
  >
    <br>
    <div class="space-y-2">
      <fieldset
        v-for="(entry, idx) in fields"
        :key="(entry as any).key"
      >
        <div
          class="
            grid grid-flow-col
            auto-cols-max
            md:auto-cols-min
            space-x-2
          "
        >
          <div
            v-for="element in field.elements"
            :key="element.key"
          >
            <div class="w-2/5 md:w-full">
              <label
                :for="`${element.key}_${idx as any}`"
                class="inline md:text-right mb-1 md:mb-0 pr-4"
              >{{ element.key }}</label>
              <VeeField
                :id="`${element.key}_${idx as any}`"
                :name="`${props.field.id}[${idx as any}].${element.key}`"
                :placeholder="element.default"
                class="
                  inline
                  bg-gray-100
                  appearance-none
                  border-0 border-gray-200
                  rounded
                  py-2
                  px-4
                  text-gray-700
                  leading-tight
                  focus:outline-none
                  focus:bg-white
                  focus:border-gray-500
                "
              />
              <ErrorMessage
                :name="`${props.field.id}[${idx as any}].${element.key}`"
                class="text-red-600"
              />
            </div>
          </div>

          <div class="w-1/5 md:w-full">
            <label
              class="
                inline
                md:text-right
                mb-1
                md:mb-0
                pr-4
                text-black
              "
            >.</label>
            <button
              type="button"
              class="
                inline-flex
                transition-colors
                duration-150
                bg-transparent
                rounded
                focus:shadow-outline
                hover:bg-red-100
              "
              @click="remove(idx as unknown as any)"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                x="0px"
                y="0px"
                width="40"
                height="40"
                viewBox="0 0 48 48"
                style="fill: #000000"
              >
                <path
                  fill="#F44336"
                  d="M21.5 4.5H26.501V43.5H21.5z"
                  transform="rotate(45.001 24 24)"
                />
                <path
                  fill="#F44336"
                  d="M21.5 4.5H26.5V43.501H21.5z"
                  transform="rotate(135.008 24 24)"
                />
              </svg>
            </button>
          </div>
        </div>
      </fieldset>

      <button
        type="button"
        class="
          inline-flex
          items-center
          h-10
          px-5
          transition-colors
          duration-150
          bg-transparent
          border-0
          rounded
          focus:shadow-outline
          hover:bg-gray-100
        "
        @click="
          push(
            (field.elements ?? []).reduce(
              (acc, element) => ((acc[element.key] = ''), acc),
              {} as Record<string, string>
            )
          )
        "
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          x="0px"
          y="0px"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          style="fill: #6b7280"
          class="w-4 h-4 mr-3 fill-current"
        >
          <path
            fill-rule="evenodd"
            d="M 11 2 L 11 11 L 2 11 L 2 13 L 11 13 L 11 22 L 13 22 L 13 13 L 22 13 L 22 11 L 13 11 L 13 2 Z"
          />
        </svg>
        <span class="md:text-right mb-1 md:mb-0 pr-4">
          Add Element
        </span>
      </button>
    </div>
  </FieldArray>
</template>

<script lang="ts" setup>
import {
  Field as VeeField,
  FieldArray,
  ErrorMessage
} from 'vee-validate'

import type { FormField } from '@/task_creation_form'

interface Props { field: FormField }
const props = defineProps<Props>()
</script>
