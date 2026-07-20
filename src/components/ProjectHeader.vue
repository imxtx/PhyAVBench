<script setup lang="ts">
const baseUrl = import.meta.env.BASE_URL
</script>

<template>
  <section class="section project-header">
    <h1 class="page-title">PhyAVBench: A Challenging Audio Physics-Sensitivity Benchmark for Physically Grounded Text-to-Audio-Video Generation</h1>

    <div class="author-block">
      <p class="author-list">
        Tianxin Xie<sup>1,2,†</sup>, Wentao Lei<sup>1,†</sup>, Kai Jiang<sup>1,†</sup>,
        Guanjie Huang<sup>1,†</sup>, Pengfei Zhang<sup>1,†</sup>, Chunhui Zhang<sup>1</sup>,
        Fengji Ma<sup>1</sup>, Haoyu He<sup>1</sup>, Han Zhang<sup>1</sup>,
        Jiangshan He<sup>1</sup>, Jinting Wang<sup>1</sup>, Linghan Fang<sup>1</sup>,
        Lufei Gao<sup>1</sup>, Orkesh Ablet<sup>1</sup>, Peihua Zhang<sup>2</sup>,
        Ruolin Hu<sup>1</sup>, Shengyu Li<sup>1</sup>, Weilin Lin<sup>1</sup>,
        Xiaoyang Feng<sup>1</sup>, Xinyue Yang<sup>1</sup>, Yan Rong<sup>1</sup>,
        Yanyun Wang<sup>1</sup>, Zihang Shao<sup>1</sup>, Zelin Zhao<sup>1</sup>,
        Chenxing Li<sup>2</sup>, Shan Yang<sup>2</sup>, Wenfu Wang<sup>2</sup>,
        Meng Yu<sup>2</sup>, Dong Yu<sup>2</sup>, Li Liu<sup>1,*</sup>
      </p>
      <p class="affiliations"><sup>1</sup>HKUST(GZ) &nbsp;&nbsp; <sup>2</sup>Tencent</p>
      <p class="author-notes">
        <span><sup>†</sup> Core contributors.</span>
        <span><sup>*</sup> Corresponding author: Li Liu</span>
      </p>
    </div>

    <img alt="PhyAVBench logo" class="project-logo" :src="`${baseUrl}images/logo.png`" />

    <nav class="project-links" aria-label="Project resources">
      <a href="https://arxiv.org/abs/2512.23994" target="_blank" rel="noopener noreferrer">Paper</a>
      <a href="https://github.com/imxtx/PhyAVBench" target="_blank" rel="noopener noreferrer">Code</a>
      <a href="https://huggingface.co/datasets/imxtx/PhyAV-Sound-11K" target="_blank" rel="noopener noreferrer">Dataset</a>
    </nav>

    <div class="summary-block">
      <h2>Abstract</h2>
      <p>
        Text-to-audio-video (T2AV) generation is central to applications such as filmmaking and world modeling. However, current models often fail to produce physically plausible sounds. Previous benchmarks in this area primarily focus on audio-video temporal synchronization, while largely overlooking explicit evaluation of audio-physics grounding, thereby limiting the study of physically plausible audio-visual generation. To address this issue, we present <strong>PhyAVBench</strong>, the first benchmark designed to systematically evaluate the audio-physics grounding capabilities of T2AV, image-to-audio-video (I2AV), and video-to-audio (V2A) models. PhyAVBench offers <strong>PhyAV-Sound-11K</strong>, a new dataset of 25.5 hours of 11,605 audible videos collected from 184 participants to ensure diversity and avoid data leakage. It contains 337 paired-prompt groups with controlled physical variations that drive sound differences, each grounded with an average of 17 videos and spanning 6 audio-physics dimensions and 41 fine-grained test points, from basic phenomena (e.g., collision) to complex effects (e.g., Helmholtz resonance). Each video includes step-by-step audio-physics reasoning, and each prompt pair is annotated with the physical factors underlying their acoustic differences. Importantly, unlike prior benchmarks that cannot measure sensitivity to underlying acoustic conditions, PhyAVBench leverages paired text prompts to evaluate this capability. We term this evaluation paradigm the Audio-Physics Sensitivity Test (APST) and introduce a novel metric, the <strong>Contrastive Physical Response Score (CPRS)</strong>, which quantifies the acoustic consistency between generated videos and their real-world counterparts. We conduct a comprehensive evaluation of 17 state-of-the-art (SOTA) models across T2AV, I2AV, and V2A tasks, along with human studies involving 74 participants, which shows a strong positive correlation with the CPRS metric. Our results reveal that even leading commercial models struggle with fundamental audio-physical phenomena, exposing a critical gap beyond audio-visual synchronization and pointing to future research directions. We hope PhyAVBench will serve as a foundation for advancing physically grounded audio-visual generation. Prompts, ground-truth, and generated video samples are available at <a href="https://phyavbench.pages.dev/">https://phyavbench.pages.dev/</a>.
      </p>
    </div>

    <div class="citation-block">
      <h2>Citation</h2>
      <pre><code>@article{xie2025phyavbench,
  title={Phyavbench: A challenging audio physics-sensitivity benchmark for physically grounded text-to-audio-video generation},
  author={Xie, Tianxin and Lei, Wentao and Jiang, Kai and Huang, Guanjie and Zhang, Pengfei and Zhang, Chunhui and Ma, Fengji and He, Haoyu and Zhang, Han and He, Jiangshan and Wang, Jinting and Fang, Linghan and Gao, Lufei and Ablet, Orkesh and Zhang, Peihua and Hu, Ruolin and Li, Shengyu and Lin, Weilin and Feng, Xiaoyang and Yang, Xinyue and Rong, Yan and Wang, Yanyun and Shao, Zihang and Zhao, Zelin and Li, Chenxing and Yang, Shan and Wang, Wenfu and Yu, Meng and Yu, Dong and Liu, Li},
  journal={arXiv preprint arXiv:2512.23994},
  year={2025}
}</code></pre>
    </div>
  </section>
</template>

<style scoped>
.project-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 1.5rem;
}

.project-header > * {
  min-width: 0;
}

.project-logo {
  display: block;
  width: 200px;
  height: auto;
  margin: 0 auto;
}

.author-block {
  display: grid;
  gap: 0.55rem;
  max-width: 1180px;
  margin: 0 auto;
  text-align: center;
}

.author-list {
  font-size: 1.05rem;
  line-height: 1.8;
}

.author-block sup {
  font-size: 0.7em;
  font-weight: 600;
}

.affiliations {
  font-size: 1rem;
  font-weight: 600;
}

.author-notes {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.25rem 1.25rem;
  color: var(--color-text);
  font-size: 0.9rem;
}

.project-links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.75rem;
}

.project-links a {
  min-width: 6.5rem;
  padding: 0.65rem 1.25rem;
  border: 1px solid hsla(160, 100%, 37%, 0.65);
  border-radius: 999px;
  background: hsla(160, 100%, 37%, 0.1);
  color: hsla(160, 100%, 32%, 1);
  font-weight: 600;
  text-align: center;
  transition:
    background-color 0.2s ease,
    border-color 0.2s ease,
    transform 0.2s ease;
}

.project-links a:hover {
  border-color: hsla(160, 100%, 37%, 1);
  background: hsla(160, 100%, 37%, 0.2);
  transform: translateY(-1px);
}

.project-links a:focus-visible {
  outline: 3px solid hsla(160, 100%, 37%, 0.3);
  outline-offset: 2px;
}

.page-title {
  margin: 0;
  font-size: 1.90rem;
  line-height: 1.15;
}

.summary-block h2,
.citation-block h2 {
  margin: 0 0 0.6rem;
  font-size: 1.25rem;
}

.summary-block p {
  margin: 0;
  line-height: 1.7;
}

.citation-block pre {
  overflow-x: auto;
  max-width: 100%;
  margin: 0;
  padding: 1.25rem;
  border: 1px solid var(--color-border);
  border-radius: 0.6rem;
  background: var(--color-background-soft);
  font-size: 0.86rem;
  line-height: 1.6;
  white-space: pre;
}

.latex-example {
  margin-top: 0.8rem;
}
</style>
