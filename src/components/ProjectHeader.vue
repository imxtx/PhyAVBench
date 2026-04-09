<template>
  <section class="section project-header">
    <h1 class="page-title">PhyAVBench: A Challenging Audio Physics-Sensitivity Benchmark for Physically Grounded Text-to-Audio-Video Generation</h1>
    <img alt="PhyAVBench logo" class="project-logo" src="/images/logo.png" />

    <div class="summary-block">
      <h2>Abstract</h2>
      <p>
        Text-to-audio-video (T2AV) generation is central to applications such as filmmaking and world modeling. However, current models often fail to produce physically plausible sounds. Previous benchmarks in this area primarily focus on audio-video temporal synchronization, while largely overlooking explicit evaluation of audio-physics grounding, thereby limiting the study of physically plausible audio-visual generation. To address this issue, we present <strong>PhyAVBench</strong>, the first benchmark designed to systematically evaluate the audio-physics grounding capabilities of T2AV, image-to-audio-video (I2AV), and video-to-audio (V2A) models. PhyAVBench offers <strong>PhyAV-Sound-11K</strong>, a new dataset of 25.5 hours of 11,605 audible videos collected from 184 participants to ensure diversity and avoid data leakage. It contains 337 paired-prompt groups with controlled physical variations that drive sound differences, each grounded with an average of 17 videos and spanning 6 audio-physics dimensions and 41 fine-grained test points, from basic phenomena (e.g., collision) to complex effects (e.g., Helmholtz resonance). Each video includes step-by-step audio-physics reasoning, and each prompt pair is annotated with the physical factors underlying their acoustic differences. Importantly, unlike prior benchmarks that cannot measure sensitivity to underlying acoustic conditions, PhyAVBench leverages paired text prompts to evaluate this capability. We term this evaluation paradigm the Audio-Physics Sensitivity Test (APST) and introduce a novel metric, the <strong>Contrastive Physical Response Score (CPRS)</strong>, which quantifies the acoustic consistency between generated videos and their real-world counterparts. We conduct a comprehensive evaluation of 17 state-of-the-art (SOTA) models across T2AV, I2AV, and V2A tasks, along with human studies involving 74 participants, which shows a strong positive correlation with the CPRS metric. Our results reveal that even leading commercial models struggle with fundamental audio-physical phenomena, exposing a critical gap beyond audio-visual synchronization and pointing to future research directions. We hope PhyAVBench will serve as a foundation for advancing physically grounded audio-visual generation. Prompts, ground-truth, and generated video samples are available at <a href="https://phyavbench.pages.dev/">https://phyavbench.pages.dev/</a>.
      </p>
    </div>
  </section>
</template>

<style scoped>
.project-header {
  display: grid;
  gap: 1.5rem;
}

.project-logo {
  display: block;
  width: 200px;
  height: auto;
  margin: 0 auto;
}

.page-title {
  margin: 0;
  font-size: 1.90rem;
  line-height: 1.15;
}

.summary-block h2 {
  margin: 0 0 0.6rem;
  font-size: 1.25rem;
}

.summary-block p {
  margin: 0;
  line-height: 1.7;
}

.latex-example {
  margin-top: 0.8rem;
}
</style>
