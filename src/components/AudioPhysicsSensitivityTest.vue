<template>
  <section class="section">
    <h2>Audio-Physics Sensitivity Test</h2>
    <figure class="apst-figure">
      <img class="apst-image" src="/images/apst.png" alt="Audio-Physics Sensitivity Test overview" />
      <figcaption>
        Figure 1. Overview of the proposed Audio-Physics Sensitivity Test (APST). APST uses paired prompts that differ by a single physical variable (e.g., material). By comparing the direction and magnitude trends of generated audio features against ground-truth ones, we calculate the Contrastive Physical Response Score (CPRS) to assess the model's understanding of real-world physics.
      </figcaption>
    </figure>
  </section>
</template>

<style scoped>
h2 {
  margin: 0 0 0.75rem;
  font-size: 1.5rem;
}

.apst-figure {
  margin: 0;
  display: grid;
  justify-items: center;
  gap: 0.9rem;
}

.apst-image {
  display: block;
  width: min(100%, 600px);
  height: auto;
}

figcaption {
  font-style: italic;
}
</style>
