<script setup lang="ts">
import { onMounted, ref } from 'vue'
import VideoTable from './VideoTable.vue'
import type { VideoGroup } from './VideoTable.vue'

type RawGroup = {
  id: string
  prompt_a: string
  prompt_b: string
  ref_video_a: string
  ref_video_b: string
}

const groups = ref<VideoGroup[]>([])
const isLoading = ref(true)
const loadError = ref('')

function parseJsonl(content: string): VideoGroup[] {
  return content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as RawGroup)
    .map((item) => ({
      id: item.id,
      promptA: item.prompt_a,
      promptB: item.prompt_b,
      gtA: item.ref_video_a,
      gtB: item.ref_video_b,
    }))
}

onMounted(async () => {
  try {
    const response = await fetch('/videos480p/10_groups.jsonl')
    if (!response.ok) {
      throw new Error(`Failed to load groups: ${response.status}`)
    }

    const text = await response.text()
    groups.value = parseJsonl(text)
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : 'Unknown error'
  } finally {
    isLoading.value = false
  }
})
</script>

<template>
  <section class="section">
    <h2>Sample Video Pairs in PhyAVBench</h2>
    <p>
      Each prompt group is grounded by an average of 17 newly recorded videos, thereby minimizing the risk of data leakage during model pre-training. The following are some sample video pairs in PhyAVBench, showing the diversity of the data.
    </p>

    <p v-if="isLoading" class="state-text">Loading video groups...</p>
    <p v-else-if="loadError" class="state-text error">{{ loadError }}</p>

    <div v-else class="video-list">
      <VideoTable v-for="group in groups" :key="group.id" :group="group" />
    </div>
  </section>
</template>

<style scoped>
h2 {
  margin: 0 0 0.75rem;
  font-size: 1.5rem;
}

p {
  margin: 0;
  line-height: 1.7;
}

.video-list {
  margin-top: 1rem;
  display: grid;
  gap: 1.6rem;
}

.state-text {
  margin-top: 0.8rem;
}

.error {
  color: #b42318;
}
</style>
