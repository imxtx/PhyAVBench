<script setup lang="ts">
import { computed } from 'vue'

type ModelEntry = {
  name: string
  folder: string
}

export type VideoGroup = {
  id: string
  promptA: string
  promptB: string
  gtA: string
  gtB: string
}

const props = defineProps<{
  group: VideoGroup
}>()

const displayId = computed(() => props.group.id.replace(/_[ab]$/, ''))

const modelEntries: ModelEntry[] = [
  { name: 'GT', folder: 'gt' },
  { name: 'Sora 2', folder: 'sora2' },
  { name: 'Wan 2.6', folder: 'wan2.6' },
  { name: 'Seedance 1.5 Pro', folder: 'seedance-1-5-pro' },
  { name: 'Kling v2.6', folder: 'klingv2-6' },
  { name: 'Veo 3.1', folder: 'veo3.1' },
  { name: 'Ovi', folder: 'ovi' },
  { name: 'LTX', folder: 'ltx' },
  { name: 'JavisDiT', folder: 'javisdit' },
  { name: 'JavisDiT++', folder: 'javisdit++' },
]

function getVideoPath(folder: string, variant: 'a' | 'b'): string {
  const fileName =
    folder === 'gt'
      ? variant === 'a'
        ? props.group.gtA
        : props.group.gtB
      : `${props.group.id}_${variant}`

  return `/videos480p/${folder}/${fileName}.mp4`
}

function getMelPath(folder: string, variant: 'a' | 'b'): string {
  const fileName =
    folder === 'gt'
      ? variant === 'a'
        ? props.group.gtA
        : props.group.gtB
      : `${props.group.id}_${variant}`

  return `/mels/${folder}/${fileName}.png`
}

function formatBoldText(text: string): string {
  return text.replace(/\*\*([^*]+)\*\*/g, '<strong class="bold-red">$1</strong>')
}
</script>

<template>
  <article class="video-group">
    <h3 class="group-id">{{ displayId }}</h3>
    <div class="video-table-wrap">
    <table class="video-table">
      <thead>
        <tr>
          <th>Prompts</th>
          <th v-for="model in modelEntries" :key="model.folder">{{ model.name }}</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="prompt-cell" rowspan="2">
            <strong>a</strong>
            <p v-html="formatBoldText(props.group.promptA)"></p>
          </td>
          <td v-for="model in modelEntries" :key="`${model.folder}-a`" class="video-cell">
            <video class="media-frame" controls preload="none" playsinline>
              <source :src="getVideoPath(model.folder, 'a')" type="video/mp4" />
            </video>
          </td>
        </tr>
        <tr>
          <td v-for="model in modelEntries" :key="`${model.folder}-a-mel`" class="video-cell">
            <img class="media-frame" :src="getMelPath(model.folder, 'a')" :alt="`${model.name} mel for prompt a`" loading="lazy" />
          </td>
        </tr>
        <tr>
          <td class="prompt-cell" rowspan="2">
            <strong>b</strong>
            <p v-html="formatBoldText(props.group.promptB)"></p>
          </td>
          <td v-for="model in modelEntries" :key="`${model.folder}-b`" class="video-cell">
            <video class="media-frame" controls preload="none" playsinline>
              <source :src="getVideoPath(model.folder, 'b')" type="video/mp4" />
            </video>
          </td>
        </tr>
        <tr>
          <td v-for="model in modelEntries" :key="`${model.folder}-b-mel`" class="video-cell">
            <img class="media-frame" :src="getMelPath(model.folder, 'b')" :alt="`${model.name} mel for prompt b`" loading="lazy" />
          </td>
        </tr>
      </tbody>
    </table>
  </div>
  </article>
</template>

<style scoped>
.video-group {
  display: grid;
  gap: 0.65rem;
}

.group-id {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
}

.video-table-wrap {
  display: block;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow-x: scroll;
  overflow-y: hidden;
  scrollbar-gutter: stable;
  -webkit-overflow-scrolling: touch;
}

.video-table {
  width: max-content;
  border-collapse: separate;
  border-spacing: 0;
}

.video-table-wrap::-webkit-scrollbar {
  height: 10px;
}

.video-table-wrap::-webkit-scrollbar-track {
  background: #e5e7eb;
  border-radius: 999px;
}

.video-table-wrap::-webkit-scrollbar-thumb {
  background: #94a3b8;
  border-radius: 999px;
}

.video-table th,
.video-table td {
  border-bottom: 1px solid #e2e8f0;
  padding: 0.7rem;
  vertical-align: middle;
  background: #ffffff;
}

.video-table thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #3498db;
  color: #ffffff;
  font-weight: 700;
  text-align: center;
  white-space: nowrap;
}

.prompt-cell {
  width: 350px;
}

.prompt-cell strong {
  display: inline-block;
  margin-bottom: 0.4rem;
  font-size: 1rem;
}

.prompt-cell p {
  margin: 0;
  line-height: 1.55;
  font-size: 0.9rem;
}

:deep(.bold-red) {
  font-weight: 700;
  color: #cc0000;
}

.media-frame {
  display: block;
  width: 200px;
  height: 150px;
}

.video-cell video {
  border-radius: 8px;
  background: #000;
  object-fit: cover;
}

.video-cell img {
  border-radius: 8px;
  border: 1px solid #dbe4ef;
  object-fit: fill;
  background: #f8fafc;
}
</style>
