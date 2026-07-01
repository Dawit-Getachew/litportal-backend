"""Shared XML fixtures for Phase 4A tests."""
from __future__ import annotations

# A small but realistic JATS document (Europe PMC fullTextXML shape).
JATS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
  <front>
    <article-meta>
      <title-group>
        <article-title>Apixaban versus warfarin in atrial fibrillation</article-title>
      </title-group>
      <permissions>
        <license license-type="open-access">
          <license-p>Distributed under <ext-link xlink:href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</ext-link>.</license-p>
        </license>
      </permissions>
      <abstract>
        <p>In ARISTOTLE, apixaban reduced major bleeding and stroke versus warfarin in atrial fibrillation [1]. The findings supported a new standard of anticoagulation care.</p>
      </abstract>
    </article-meta>
  </front>
  <body>
    <sec sec-type="intro"><title>Introduction</title>
      <p>Atrial fibrillation is the most common sustained arrhythmia and carries a fivefold higher risk of stroke.</p>
    </sec>
    <sec sec-type="methods"><title>Methods</title>
      <p>This was a randomized, double-blind, double-dummy trial enrolling 18201 participants comparing apixaban with warfarin.</p>
    </sec>
    <sec sec-type="results"><title>Results</title>
      <p>Apixaban reduced the primary endpoint of stroke or systemic embolism by 21% versus warfarin (HR 0.79). Major bleeding fell by 31% (Smith et al, 2011).</p>
    </sec>
    <sec sec-type="discussion"><title>Discussion</title>
      <p>The results extend prior evidence on direct oral anticoagulants and inform clinical guideline updates.</p>
    </sec>
    <sec sec-type="conclusion"><title>Conclusions</title>
      <p>Apixaban was superior to warfarin for stroke prevention and reduced major bleeding.</p>
    </sec>
  </body>
  <back>
    <ack><p>We thank the study coordinators.</p></ack>
    <ref-list>
      <ref id="r1"><label>1</label><element-citation publication-type="journal"><source>NEJM</source></element-citation></ref>
    </ref-list>
  </back>
</article>
"""


JATS_TABLES_FIGS = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>Tables and figures sample</article-title></title-group>
      <permissions>
        <license license-type="open-access">
          <license-p><ext-link xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</ext-link></license-p>
        </license>
      </permissions>
      <abstract><p>Short abstract paragraph used to satisfy the minimum section length requirement when added together.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec sec-type="results">
      <title>Results</title>
      <p>Primary outcome favored treatment with statistically significant reduction in events as detailed in the tables below.</p>
      <table-wrap>
        <caption><p>Table 1. Primary outcome by group with hazard ratios and 95% confidence intervals.</p></caption>
        <table>
          <thead><tr><th>Group</th><th>Events</th><th>HR</th></tr></thead>
          <tbody>
            <tr><td>Apixaban</td><td>212</td><td>0.79</td></tr>
            <tr><td>Warfarin</td><td>265</td><td>1.00</td></tr>
          </tbody>
        </table>
      </table-wrap>
      <fig>
        <caption><p>Figure 1. Kaplan-Meier curves showing stroke-free survival across treatment arms.</p></caption>
      </fig>
    </sec>
  </body>
</article>
"""


JATS_NO_LICENSE = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <article-meta>
      <title-group><article-title>No license metadata</article-title></title-group>
      <abstract><p>This article has no license metadata in the JATS permissions block and the gate should refuse ingest.</p></abstract>
    </article-meta>
  </front>
  <body><sec><title>Body</title><p>Some text content for the body section that is long enough to survive the minimum-length filter.</p></sec></body>
</article>
"""


# A small BioC document (NCBI BioC_xml/{PMCID}/unicode shape).
BIOC_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <document>
    <id>PMC1234567</id>
    <passage>
      <infon key="type">title</infon>
      <infon key="section_type">title_1</infon>
      <text>Empagliflozin in heart failure with reduced ejection fraction</text>
    </passage>
    <passage>
      <infon key="type">abstract</infon>
      <infon key="section_type">abstract</infon>
      <text>EMPEROR-Reduced showed empagliflozin reduced the combined risk of cardiovascular death or hospitalization for heart failure versus placebo across diabetes status.</text>
    </passage>
    <passage>
      <infon key="type">methods</infon>
      <infon key="section_type">methods</infon>
      <text>Randomized double-blind placebo-controlled trial in 3730 patients with class II-IV HF and an LVEF of 40% or less.</text>
    </passage>
    <passage>
      <infon key="type">results</infon>
      <infon key="section_type">results</infon>
      <text>Empagliflozin reduced the primary outcome by 25% (HR 0.75, 95% CI 0.65 to 0.86, p&lt;0.001) with consistent effects across pre-specified subgroups.</text>
    </passage>
    <passage>
      <infon key="type">discuss</infon>
      <infon key="section_type">discussion</infon>
      <text>The benefit was independent of diabetes status, consistent with prior SGLT2 inhibitor trials and supporting broader heart failure indications.</text>
    </passage>
    <passage>
      <infon key="type">concl</infon>
      <infon key="section_type">conclusion</infon>
      <text>Empagliflozin reduces heart-failure events in patients with reduced ejection fraction.</text>
    </passage>
    <passage>
      <infon key="type">ref</infon>
      <infon key="section_type">references</infon>
      <text>1. Anker SD, et al. EMPEROR-Reduced. N Engl J Med 2020.</text>
    </passage>
    <passage>
      <infon key="license">https://creativecommons.org/licenses/by/4.0/</infon>
      <text></text>
    </passage>
  </document>
</collection>
"""


PMC_OA_SVC_OK = """<?xml version="1.0"?>
<OA>
  <records>
    <record id="oai:pubmedcentral.nih.gov:1234567" license="CC BY">
      <link format="pdf" updated="2025-01-01" href="ftp://example/pdf"/>
    </record>
  </records>
</OA>
"""


PMC_OA_SVC_NOT_OA = """<?xml version="1.0"?>
<OA>
  <error code="idIsNotOpenAccess">Article not in PMC OA subset.</error>
</OA>
"""


PMC_OA_SVC_UNKNOWN_LIC = """<?xml version="1.0"?>
<OA>
  <records>
    <record id="oai:pubmedcentral.nih.gov:9999999">
      <link format="pdf" updated="2025-01-01" href="ftp://example/pdf"/>
    </record>
  </records>
</OA>
"""
