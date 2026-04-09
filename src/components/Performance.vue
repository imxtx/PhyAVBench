<template>
  <section class="section">
    <h2>Performance Comparison</h2>
    <figure class="rador-figure">
      <img class="rador-image" src="/images/rador.png" alt="Audio-Physics Sensitivity Test overview" />
      <figcaption>
        Figure 2. The audio-physics grounding performance of SOTA T2AV, I2AV, and V2A models measured by CPRS.
      </figcaption>
    </figure>
  </section>
</template>

<style scoped>
h2 {
  margin: 0 0 0.75rem;
  font-size: 1.5rem;
}

.rador-figure {
  margin: 0;
  display: grid;
  justify-items: center;
  gap: 0.9rem;
}

.rador-image {
  display: block;
  width: 100%;
  height: auto;
}

figcaption {
  font-style: italic;
}
</style>
