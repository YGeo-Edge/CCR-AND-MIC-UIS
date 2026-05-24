# MIC2 VLM Classifier — GAP Analysis for v2

**Based on:** training evaluation (7,846 test images), nn_mlcs scan (3,540 unseen images), visual inspection, threshold analysis.

---

## 1. Class-Level Weaknesses

### GAP-01 · financial_scam — worst single class (P1)
- **FN: 18/36** (50% of all missed malicious come from this class)
- **FP: 11/18** (61% of all false alarms are financial_scam)
- **Block rate in production scan: 28%** — 486 of 675 detections go to review
- Threshold must be set at 0.999 to achieve FPR < 0.1%, meaning the model is rarely confident enough to auto-block
- **Root cause:** huge visual diversity in financial scam pages (fake news articles, investment lures, health supplement, lottery — all look different)
- **Fix:** expand training data with subcategories; consider splitting into sub-classes (fake_news_finance, health_supplement_scam, lottery_scam)

### GAP-02 · forced_notification — almost no auto-blocks (P1)
- **Block rate: 5%** — 311 of 326 detections go to review
- 151 detections have conf ≥ 0.90 but still can't reach 0.999 threshold
- 3 FNs in test set
- **Root cause:** visual appearance is diverse (any page can trigger a push-notification popup overlay); the model is uncertain because the "malicious" signal is a small popup on top of innocent-looking content
- **Fix:** more training examples, especially with the popup visible; consider a two-crop strategy (full page + popup region)

### GAP-03 · fake_av — false positives on legitimate AV sites (P1)
- Confirmed FP: **real www.mcafee.com website** classified as fake_av at 0.999 conf
- Block rate in scan: 3% (1 of 37)
- **Root cause:** training set contains only fake AV pages; model never saw a legitimate AV product page, so it can't distinguish real from fake
- **Fix:** add hard-negative examples from legitimate AV vendors (McAfee, Norton, Bitdefender, Kaspersky, ESET) to training data as Benign

### GAP-04 · malicious_extension — too low confidence (P2)
- All 40 review-bucket items have conf < 0.80 (below threshold)
- Block rate: 7%
- Confirmed ambiguity: Ghostery ad blocker (legitimate extension) on suspicious domain
- **Root cause:** hard to distinguish legitimate browser extension promotion from malicious push
- **Fix:** add hard-negatives from official extension stores (Chrome Web Store, Firefox Add-ons); add more malicious extension examples from wild

### GAP-05 · fake_updates — zero auto-blocks (P2)
- 9 detections in scan, all in review; max conf 0.973, threshold 0.999
- 1 FN in test set
- **Root cause:** rare class — low training sample count limits confidence calibration
- **Fix:** collect more fake_updates examples; threshold can likely be lowered with more data

### GAP-06 · fake_av — zero auto-blocks in wild (P2)
- 36 review detections, only 1 block, threshold 0.999
- 1 FN in test set
- **Fix:** same as GAP-03 — more data + hard-negatives will allow threshold to be lowered

---

## 2. Threshold & Confidence Calibration

### GAP-07 · Review bucket is operationally too large (P1)
- In the nn_mlcs scan: **1,272 of 1,800 malicious (71%) require human review**
- Five classes (financial_scam, forced_notification, fake_av, fake_updates, misleading_offers) have thresholds ≥ 0.990 because the model isn't confident enough to auto-block safely
- **Fix options:**
  - More training data per class to push confidence higher
  - Apply **temperature scaling** (post-hoc calibration on validation set) to better align confidence with accuracy
  - Introduce a **confidence band**: e.g., conf < 0.50 = likely benign, pass; 0.50–0.90 = weak signal, log only; > 0.90 = review; per-class threshold = block

### GAP-08 · Thresholds derived from test set — need independent calibration set (P2)
- Current thresholds were computed on the same test split used for final reporting
- This risks slight overfitting of threshold values
- **Fix:** hold out a separate calibration split (e.g., 5%) for threshold tuning, keep test set clean

---

## 3. Unlabeled / Mislabeled Data

### GAP-09 · 1,740 images in nn_mlcs predicted BENIGN (P1)
- The nn_mlcs folder is labeled "malicious" by source system, but model predicts 49.2% as BENIGN with avg confidence 0.959
- Either: (a) these images are genuinely benign and the source labeling was wrong, or (b) they represent new attack patterns the model hasn't seen
- **56 low-confidence Benign cases** (conf < 0.60) where second-best class is forced_notification, fake_updates, or malicious_extension — these are the most suspicious
- **Fix:** human review of the 56 low-conf cases first; then sample-review the 1,729 high-conf Benign cases; relabel and add to training

### GAP-10 · nn_mlcs may contain new attack patterns not in training (P2)
- Domains like `gokilbabe88.online`, `getgaysexhot.info`, `rt.runetki5.com` are suspicious but model is uncertain
- Some may be adult content / social engineering pages not covered by current 13 classes
- **Fix:** audit unlabeled predictions, identify new pattern clusters, consider adding new classes (e.g., adult_social_engineering)

---

## 4. Training Data Gaps

### GAP-11 · No hard-negatives from legitimate brands (P1)
- Confirmed: real McAfee → fake_av FP
- Risk: real Google Play Store → fake_appstore FP; real Norton/Avast → fake_av FP
- **Fix:** add 200–500 screenshots of legitimate brand pages per affected class as hard-negative Benign examples

### GAP-12 · Class data imbalance (P2)
- financial_scam dominates errors (largest class, most FP and FN)
- fake_updates, suspicious_vpn, fake_av have very few examples → model uncertainty
- **Fix:** oversample or augment small classes; apply class-weighted loss in training

### GAP-13 · Multilingual content (P3)
- Seen pages in Polish, Russian, Japanese, Filipino/Tagalog, Turkish
- Model handles vision well but text-heavy scam pages in non-English may confuse it
- **Fix:** ensure training data covers multilingual pages proportionally; consider OCR-augmented features

---

## 5. Model Architecture

### GAP-14 · Vision-only: LLM discarded (P2)
- Current model uses only InternViT-300M (vision encoder); the LLM head is thrown away
- Text on the page (URLs, headlines, button labels) is only visible through pixel patterns, not semantic understanding
- **Fix for v2:** keep the LLM component and use it for text-based reasoning about the page content; or add an OCR + text-classifier branch

### GAP-15 · Single-crop inference (P3)
- Model sees the full-page screenshot resized to 448×448
- Forced notification popups, small browser dialogs, and extension install prompts are tiny relative to the full page and get downscaled
- **Fix:** multi-scale or multi-crop inference: full page + crop of key regions (top-third, center popup detection)

### GAP-16 · No temporal / URL signal (P3)
- Model only uses the visual screenshot; URL/domain features are not used
- Many correct detections were from obviously suspicious domains (`.xyz`, `.pro`, random hashes) — these could be zero-cost signals
- **Fix:** add URL/domain features as a lightweight second-stage signal or pre-filter

---

## 6. Evaluation & Process

### GAP-17 · No regression testing framework (P2)
- Each version is evaluated on a new test split; no fixed "golden test set" for cross-version comparison
- **Fix:** freeze a held-out golden test set (~2,000 images) that is never trained on, used only for version-to-version comparison

### GAP-18 · No online feedback loop (P2)
- FP/FN from production (like this nn_mlcs scan) are not automatically fed back into training
- **Fix:** build a labeling pipeline: scan → human review queue → label → retrain cycle

---

## Priority Summary

| # | Gap | Priority | Effort | Impact |
|---|-----|----------|--------|--------|
| GAP-01 | financial_scam diversity | P1 | High | Very High |
| GAP-02 | forced_notification low confidence | P1 | Medium | High |
| GAP-03 | fake_av FP on real AV sites | P1 | Low | High |
| GAP-07 | Review bucket too large (71%) | P1 | Medium | Very High |
| GAP-09 | 1,740 unverified benign in nn_mlcs | P1 | Medium | High |
| GAP-11 | No hard-negatives from real brands | P1 | Low | High |
| GAP-04 | malicious_extension low conf | P2 | Medium | Medium |
| GAP-05 | fake_updates near-zero blocks | P2 | Low | Medium |
| GAP-06 | fake_av near-zero blocks | P2 | Low | Medium |
| GAP-08 | Threshold calibration on test set | P2 | Low | Medium |
| GAP-10 | New attack patterns in nn_mlcs | P2 | High | Medium |
| GAP-12 | Class data imbalance | P2 | Medium | Medium |
| GAP-14 | Vision-only, LLM discarded | P2 | High | High |
| GAP-17 | No golden regression test set | P2 | Low | Medium |
| GAP-18 | No production feedback loop | P2 | High | High |
| GAP-13 | Multilingual coverage | P3 | Medium | Low |
| GAP-15 | Single-crop, misses small popups | P3 | Medium | Low |
| GAP-16 | No URL/domain signal | P3 | Low | Low |

---

## Quick Wins for v2 (low effort, high impact)

1. **Add ~500 real AV/brand screenshots as hard-negative Benign** → fixes GAP-03, reduces fake_av FP immediately
2. **Human review the 56 low-conf nn_mlcs cases + relabel** → cheap data quality fix
3. **Freeze a golden test set now** → enables proper v1→v2 regression comparison
4. **Temperature scaling on validation set** → better calibrated confidence, reduces review bucket size without retraining
