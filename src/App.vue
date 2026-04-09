<script setup lang="ts">
import ProjectHeader from './components/ProjectHeader.vue'
import AudioPhysicsSensitivityTest from './components/AudioPhysicsSensitivityTest.vue'
import ComparisonWithExistingBenchmarks from './components/ComparisonWithExistingBenchmarks.vue'
import DataCurationPipeline from './components/DataCurationPipeline.vue'
import SampleVideoPairsInPhyAVBench from './components/SampleVideoPairsInPhyAVBench.vue'
import Performance from './components/Performance.vue'
</script>

<template>
  <main class="project-page">
    <ProjectHeader />
    <AudioPhysicsSensitivityTest />
    <Performance />
    <ComparisonWithExistingBenchmarks />
    <DataCurationPipeline />
    <SampleVideoPairsInPhyAVBench />
  </main>
</template>

<style scoped>
.project-page {
  display: grid;
  gap: 3rem;
}

.project-page :deep(.section) {
  min-width: 0;
  padding: 1.4rem 1.2rem;
  border: 1px solid var(--color-border);
  border-radius: 14px;
  background: #ffffff;
}

@media (min-width: 960px) {
  .project-page {
    gap: 3.5rem;
  }

  .project-page :deep(.section) {
    padding: 2rem;
  }
}
</style>
