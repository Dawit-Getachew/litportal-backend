"""The PubMed XML parser must take the article's OWN identifiers.

PubmedData/ReferenceList carries the PMIDs/DOIs/PMCIDs of every paper the
article cites. Walking the whole record stamped the last cited paper's DOI and
PMCID on the article (live check 2026-10-03: 70/150 wrong DOIs, 108/150 wrong
PMCIDs on a septic-shock steroid search), so full text could be fetched for a
different paper.
"""
from __future__ import annotations

from services.connectors.pubmed import PubMedConnector

XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID Version="1">34605781</PMID>
   <Article>
    <ELocationID EIdType="doi" ValidYN="Y">10.1097/CCM.0000000000005337</ELocationID>
    <Abstract><AbstractText Label="RATIONALE">Own abstract.</AbstractText></Abstract>
    <Language>eng</Language>
    <PublicationTypeList><PublicationType>Practice Guideline</PublicationType></PublicationTypeList>
   </Article>
   <OtherAbstract Language="spa"><AbstractText>Resumen.</AbstractText></OtherAbstract>
   <CommentsCorrectionsList><CommentsCorrections RefType="CommentIn"><PMID>35000001</PMID></CommentsCorrections></CommentsCorrectionsList>
  </MedlineCitation>
  <PubmedData>
   <ArticleIdList>
    <ArticleId IdType="pubmed">34605781</ArticleId>
    <ArticleId IdType="doi">10.1097/CCM.0000000000005337</ArticleId>
   </ArticleIdList>
   <ReferenceList>
    <Reference><Citation>Cited paper A</Citation>
     <ArticleIdList><ArticleId IdType="doi">10.1186/s13613-017-0241-1</ArticleId><ArticleId IdType="pmc">PMC5318305</ArticleId></ArticleIdList></Reference>
    <Reference><Citation>Cited paper B</Citation>
     <ArticleIdList><ArticleId IdType="doi">10.1155/2014/698265</ArticleId></ArticleIdList></Reference>
   </ReferenceList>
  </PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation>
   <PMID Version="1">40000002</PMID>
   <Article><ELocationID EIdType="doi" ValidYN="Y">10.1000/eloc-only</ELocationID></Article>
  </MedlineCitation>
  <PubmedData>
   <ArticleIdList><ArticleId IdType="pubmed">40000002</ArticleId><ArticleId IdType="pmc">PMC1234567</ArticleId></ArticleIdList>
   <ReferenceList><Reference><ArticleIdList><ArticleId IdType="pmc">PMC7654321</ArticleId></ArticleIdList></Reference></ReferenceList>
  </PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""


def test_ids_come_from_the_article_not_its_references():
    out = PubMedConnector._parse_efetch_xml(XML)
    guideline = out["34605781"]
    assert guideline["doi_xml"] == "10.1097/CCM.0000000000005337"
    assert guideline["pmcid_xml"] is None  # its references have PMCIDs; it does not
    assert guideline["abstractText_xml"] == "RATIONALE: Own abstract."  # not the Spanish OtherAbstract
    assert guideline["pubTypeList_xml"] == ["Practice Guideline"]
    other = out["40000002"]
    assert other["pmcid_xml"] == "PMC1234567"
    assert other["doi_xml"] == "10.1000/eloc-only"  # ELocationID fallback
    assert "35000001" not in out  # a CommentsCorrections PMID is not a record
