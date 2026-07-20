<template>
  <section class="section">
    <h2>Comparison with Existing Benchmarks</h2>
    <p>
      This section is ready for comparison tables and commentary against prior benchmarks.
    </p>
    <table class="comparison-table">
      <thead>
        <tr>
          <th rowspan="2">Benchmark</th>
          <th colspan="6">Acoustic Physical Dimensions</th>
          <th rowspan="2">Controlled<br />Setting</th>
          <th rowspan="2">Newly<br />Collected</th>
          <th rowspan="2">#GT Videos<br />per Prompt</th>
          <th rowspan="2">Evaluation<br />Metric</th>
        </tr>
        <tr>
          <th>Sound Source<br />Mechanism</th>
          <th>Fluid &amp;<br />Aerodynamics</th>
          <th>Sound<br />Propagation</th>
          <th>Observer<br />Physics</th>
          <th>Time &amp;<br />Causality</th>
          <th>Complex<br />Coupling</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>TAVGBench</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td>1</td>
          <td>AV-Align</td>
        </tr>
        <tr>
          <td>SAVGBench</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td>-</td>
          <td>AV&amp;Spatial-Align</td>
        </tr>
        <tr>
          <td>Verse-Bench</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell check-yes">✓</td>
          <td>1</td>
          <td>AV-Align</td>
        </tr>
        <tr>
          <td>JavisBench</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-yes">✓ (partial)</td>
          <td>1</td>
          <td>AV-Align</td>
        </tr>
        <tr>
          <td>VABench</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td>0</td>
          <td>AV&amp;Stereo Align</td>
        </tr>
        <tr>
          <td>T2AV-Compass</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td class="check-cell">✗</td>
          <td>1</td>
          <td>AV-Align</td>
        </tr>
        <tr class="phyavbench-row">
          <td><strong>PhyAVBench (Ours)</strong></td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td class="check-cell check-yes">✓</td>
          <td>2~40</td>
          <td>AV-Align &amp; Physics Sensitivity</td>
        </tr>
      </tbody>
    </table>
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

.comparison-table {
  width: 100%;
  margin-top: 1rem;
  border-collapse: separate;
  border-spacing: 0;
  overflow: hidden;
  border-radius: 10px;
  font-size: 0.92rem;
}

.comparison-table th,
.comparison-table td {
  border-top: 0;
  border-bottom: 1px solid #d9d9d9;
  border-left: 0;
  border-right: 0;
  padding: 0.55rem 0.45rem;
  text-align: center;
  vertical-align: middle;
}

.comparison-table thead th {
  background-color: #3498db;
  color: white;
  font-weight: 700;
}

.comparison-table thead tr:first-child th:first-child {
  border-top-left-radius: 10px;
}

.comparison-table thead tr:first-child th:last-child {
  border-top-right-radius: 10px;
}

.comparison-table tbody td {
  background: #ffffff;
}

.comparison-table tbody td:first-child {
  text-align: left;
}

.check-cell {
  color: #d93025;
  font-weight: 700;
}

.check-yes {
  color: #1a7f37;
  font-weight: 700;
}

.comparison-table .phyavbench-row td {
  background: #fff5cc;
}

.comparison-table tbody tr:hover td {
  background: #eef6ff;
}

.comparison-table tbody tr.phyavbench-row:hover td {
  background: #fff5cc;
}
</style>
