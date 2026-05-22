# Shakespeare System Evaluation Comments (15-Question Suite)

This document provides a comprehensive, question-by-question comparative evaluation of the **Baseline System** (Llama-3.2-3B-Instruct with LoRA adapter disabled) and the **Improved RAG System** (Base model + FAISS Vector DB Retrieval + Shakespeare LoRA adapter), as recorded in `results/evaluation_results.csv`.

---

## 1. Overall System Scorecard

| Evaluation Dimension | Baseline System | Improved RAG System | Diagnostic Commentary |
| :--- | :---: | :---: | :--- |
| **Factual Accuracy** | 5.5 / 10 | 7.0 / 10 | RAG significantly improves factual correctness on family identities and feuds, but both systems suffer from hallucinations in complex causal-chain plot reasoning |
| **Text Grounding** | 0.5 / 10 | 8.5 / 10 | RAG excels at weaving exact Shakespearean quotes into responses; Baseline relies entirely on parametric memory with no textual citations |
| **Robustness** | 8.0 / 10 | 7.5 / 10 | Both systems detect cross-play hallucination traps, but RAG introduced new factual errors while answering Q10 |
| **Constraint Following** | 6.5 / 10 | 4.5 / 10 | RAG severely scrambles plot details when asked to explain content to a child; Baseline maintains better tone and accuracy under this constraint |
| **Style Generation Quality** | 6.0 / 10 | 8.5 / 10 | RAG produced an outstanding Early Modern English prose response, successfully mimicking authentic Shakespearean register |

---

## 2. Question-by-Question Detailed Review

---

### Q1: "Who is Hamlet?"

- **Expected Focus:** Hamlet, character concept
- **Retrieved Passage Quality:** Passages from Act 2 Scene 2, Act 3 Scene 2, Act 4 Scene 1, and Act 5 Scene 1 — all behavioural dialogue snippets showing Hamlet interacting with others. No passage directly describes his identity or background.

#### Baseline Response
> *"Hamlet is a fictional character and the main protagonist of a tragic play... He is a prince of Denmark and the son of the king of Denmark and Norway, King Hamlet."*

✅ **Assessment: Correct and comprehensive.** Accurately identifies Hamlet's role as Prince of Denmark, his father's murder, his mother's hasty remarriage, and his philosophical hesitation. A solid character-concept answer.

#### RAG Response
> *"Hamlet is the main character and the Prince of Denmark, holding the position of King. He is the son of King Claudius and Queen Gertrude, and the brother of Laertes."*

❌ **Assessment: Contains three severe factual errors.**
1. **Error 1 (Fatal):** Claims Hamlet holds "the position of King" — Hamlet **never ascends the throne** in the play. The throne is usurped by Claudius.
2. **Error 2 (Fatal):** Claims he is "the son of King Claudius" — Claudius is Hamlet's **uncle and step-father**, not his biological father. His father is the deceased King Hamlet.
3. **Error 3:** Claims Laertes is his brother — Laertes is the son of Polonius and Hamlet's **antagonist** in the final duel; they share no fraternal bond.

**🏆 Winner: Baseline.** The retrieved passages failed to provide identity-relevant context, causing the RAG model to fall back on severely distorted parametric memory, producing multiple fundamental errors.

---

### Q2: "What is the role of Lady Macbeth?"

- **Expected Focus:** Macbeth, character concept
- **Retrieved Passage Quality:** Excellent. Passages span Act 1 Scene 5 ("be the serpent under it"), Act 3 Scene 2 ("Nought's had, all's spent"), and Act 5 Scene 1 ("Out, damned spot!") — covering Lady Macbeth's full dramatic arc from conspirator to psychological ruin.

#### Baseline Response
> *"Initially, Lady Macbeth is portrayed as a strong-willed and ambitious woman who goads her husband into committing the murder. She is the driving force behind the plot..."*

✅ **Assessment: Correct.** Accurately describes her role as the primary instigator, her manipulation of Macbeth, and her eventual guilt-driven breakdown. Response is truncated but covers the core content.

#### RAG Response
> *"Lady Macbeth is a pivotal character... the driving force behind Macbeth's decision to murder King Duncan, goading him into committing the heinous act. However, her own guilt and paranoia soon consume her, as she becomes fixated on erasing the 'spot' of blood from her hands..."*

✅ **Assessment: Outstanding and well-grounded.** Accurately captures her dual character arc, and successfully integrates the retrieved Act 5 Scene 1 sleepwalking scene into the analysis. Clear structure, no hallucinations.

**🏆 Winner: RAG.** More precise, textually grounded, and hallucination-free.

---

### Q3: "Why does Macbeth kill Duncan?"

- **Expected Focus:** Macbeth, contextual motivation
- **Retrieved Passage Quality:** Problematic. Retrieval focused on Act 2 Scene 2 (post-murder terror), Act 2 Scene 3 (discovery of the body), and Act 3 Scene 6 (retrospective discussion). **Critical Act 1 passages establishing the witches' prophecy and Lady Macbeth's goading are entirely absent**, which is the root cause of the RAG system's weaker response.

#### Baseline Response
> *"Initially, Macbeth had encountered a prophecy made by the three witches that foretells his rise to the throne... Lady Macbeth's goading and manipulation ultimately push Macbeth to commit the murder."*

✅ **Assessment: Correct and complete.** Covers all three motivational pillars: personal ambition, the witches' prophecy, and Lady Macbeth's pressure — a textbook literary analysis answer.

#### RAG Response
> *"Macbeth murders King Duncan... in a premeditated act of regicide... driven by his own ambition and spurred on by the prophecies of the three witches..."*

✅ **Assessment: Factually correct, but lacking depth.** Despite the absence of supporting retrieval passages, RAG correctly identifies ambition and witches' prophecy as motivations. However, it omits Lady Macbeth's goading role, which is a significant gap. The previous `comment.md` recorded this as a "Retrieval Failure" where RAG returned no answer — this contradicts the actual CSV content, which shows a substantive (if incomplete) response.

**🏆 Winner: Baseline** (more comprehensive, explicitly covers Lady Macbeth's role), though RAG's response is not a failure.

---

### Q4: "Why does Hamlet delay taking revenge?"

- **Expected Focus:** Hamlet, contextual motivation
- **Retrieved Passage Quality:** Excellent. Passages include Act 1 Scene 4 (the ghost's appearance), Act 3 Scene 3 (Hamlet refuses to kill Claudius at prayer), and Act 4 Scene 4 (the soliloquy reflecting on his delay) — all directly relevant to the theme of procrastination.

#### Baseline Response
> *"Hamlet is torn between his desire for revenge and his moral principles... Fear of being wrong... Lack of conclusive evidence..."* (truncated)

✅ **Assessment: Correct.** Addresses Hamlet's moral dilemma, his uncertainty about the ghost's reliability, and his fear of acting wrongly. A standard but valid literary interpretation.

#### RAG Response
> *"Hamlet's decision to delay taking revenge is motivated by his concern that acting impulsively... would be ineffective... likening it to a 'spendthrift sigh' that 'hurts by easing'..."*

✅ **Assessment: Correct and textually grounded.** Successfully integrates the retrieved Act 4 Scene 7 line ("spendthrift sigh") into the analysis. One minor note: this line is spoken by **Claudius to Laertes**, not by Hamlet himself — RAG slightly misattributes the speech context, though the overall thematic interpretation remains valid.

**🏆 Winner: Draw (RAG slightly superior due to textual grounding).**

---

### Q5: "Why is Juliet conflicted after meeting Romeo?"

- **Expected Focus:** Romeo and Juliet, contextual motivation
- **Retrieved Passage Quality:** Good. Includes the key Act 1 Scene 5 lines — "My only love, sprung from my only hate!" — directly capturing Juliet's moment of conflicted discovery.

#### Baseline Response
> *"As a member of the **Montague** family, Juliet is expected to marry a member of the **Capulet** family..."*

❌ **Assessment: Catastrophic family swap.** Juliet belongs to the **Capulet** family; Romeo belongs to the **Montague** family. The Baseline reverses the core family identities — a fundamental, disqualifying factual error.

#### RAG Response
> *"Juliet is torn between her love for Romeo, a member of her sworn enemy family, the **Montagues**, and her loyalty to her own family... the **Capulets**..."*

✅ **Assessment: Factually correct on core identities.** Accurately identifies Juliet as a Capulet and Romeo as a Montague, and explains the conflict between romantic love and family loyalty. Content is solid and grounded in the retrieved text.

**🏆 Winner: RAG.** Baseline commits a foundational factual reversal; RAG correctly identifies the core family dynamic.

---

### Q6: "What is the conflict between the Montagues and the Capulets?"

- **Expected Focus:** Romeo and Juliet, family feud
- **Retrieved Passage Quality:** Excellent. Includes Act 1 Scene 1 with Tybalt's line ("I hate the word as I hate hell, all Montagues") and the Prince's famous admonishment ("Three civil brawls... have thrice disturb'd the quiet of our streets"), cutting directly to the heart of the feud.

#### Baseline Response
> *"...the animosity and hatred between the two families are fueled by their long-standing rivalry and a sense of pride and loyalty..."* (truncated)

✅ **Assessment: Correct but generic.** Describes the existence of the feud and the role of pride and honour, but is highly generalised with no textual specifics or cited incidents.

#### RAG Response
> *"...a violent confrontation between their members... the Prince of Verona has stepped in, issuing a stern warning... any further disturbances will be met with severe punishment..."*

✅ **Assessment: Correct and well-grounded.** Directly reflects the retrieved Act 1 Scene 1 content — the street brawl and the Prince's intervention — closely tied to the original text.

**🏆 Winner: RAG.** Grounded in specific textual events rather than generic commentary.

---

### Q7: "Who is Banquo, and what is his relationship with Macbeth?"

- **Expected Focus:** Macbeth, character relationship
- **Retrieved Passage Quality:** Rich. Includes Act 1 Scene 3 (Banquo and Macbeth encounter the witches together) and Act 3 Scene 1 (Macbeth's jealousy and fear of Banquo, his monologue planning the murder). Ample material to answer this question — yet the RAG model failed to utilise it properly.

#### Baseline Response
> *"Banquo is a key character and a Scottish nobleman. He is the Thane of Cawdor and the King of Scotland's most trusted advisor..."*

⚠️ **Assessment: Mostly correct with one notable error.** Banquo is **not** the Thane of Cawdor — that title belongs to Macbeth. However, the characterisation of their relationship as complex, with Banquo serving as a moral foil to Macbeth, is a reasonable literary interpretation.

#### RAG Response
> *"Banquo is a significant character... serving as a friend and fellow general to the titular character, Macbeth. He is also the son of King Duncan..."*

❌ **Assessment: Contains a severe hallucination.** Banquo has **no familial relationship with King Duncan** whatsoever. Duncan's sons are Malcolm and Donalbain. Banquo is an independent Scottish general. Claiming he is Duncan's son is a fundamental factual fabrication not supported by any retrieved passage.

**🏆 Winner: Baseline.** Although Baseline incorrectly assigns the Thane of Cawdor title to Banquo, RAG's error — inventing a father-son relationship between Banquo and Duncan — is far more severe and disqualifying.

---

### Q8: "What role does the poison play in the final scene of Romeo and Juliet?"

- **Expected Focus:** Romeo and Juliet, plot details
- **Retrieved Passage Quality:** Moderate. Includes Act 4 Scene 3 (Juliet's soliloquy before taking the potion), Act 5 Scene 1 (Romeo purchasing poison from the apothecary), and Act 5 Scene 3 fragments. Covers relevant scenes but the causal complexity proved too much for both systems.

#### Baseline Response
> *"...Romeo took a fake sleeping draught to appear dead and Juliet took actual poison to kill herself..."*

❌ **Assessment: Severe plot confusion.** The actual sequence is: **Juliet** drinks Friar Lawrence's **sleeping potion** (fake death); **Romeo** buys **real poison** from an apothecary and drinks it in Juliet's tomb; **Juliet** wakes, finds Romeo dead, and kills herself with a **dagger** — not poison. Baseline completely inverts the roles of who uses which substance.

#### RAG Response
> *"Romeo purchases the poison from an apothecary, intending to use it to end his life in Juliet's tomb... The poison is described as being particularly devastating to men's souls..."*

⚠️ **Assessment: Partially correct with key omissions and a contextual misreading.** Correctly states Romeo purchased the poison from the apothecary for suicide — this is accurate. However, RAG omits Juliet's subsequent awakening and death by dagger, leaving the narrative incomplete. The phrase "devastating to men's souls" is Romeo's metaphorical lament about **gold** as a poison ("There is thy gold — worse poison to men's souls"), which RAG misapplies as a literal description of the physical poison.

**🏆 Winner: RAG (relatively)**, though both systems fail on this question. RAG is closer to correct on Romeo's actions; Baseline fully reverses character roles.

---

### Q9: "Explain in a Shakespearean style (under 150 words): How does Macbeth feel after seeing the ghost?"

- **Expected Focus:** Macbeth, style generation
- **Retrieved Passage Quality:** Excellent. Core Act 3 Scene 4 passages retrieved — including Macbeth's confrontation with Banquo's ghost ("Hence, horrible shadow! Unreal mockery, hence!") — providing ideal material for stylistic imitation.

#### Baseline Response
> *"O, woe is Macbeth, whose soul doth tremble with fear, / As apparitions rise, and specters draw near..."* (poetic form)

✅ **Assessment: Stylistically adequate, but generic.** Uses Early Modern English markers ("doth", "hath") and adopts a poetic format. However, the content is surface-level — a generic expression of fear — without capturing Macbeth's specific internal contradiction (defiant bravado mixed with paralysing terror). May slightly exceed the 150-word limit.

#### RAG Response
> *"Fair reader, thou dost inquire what passeth in that troubled mind of Macbeth when he beholdeth the apparition of his own conscience? Marry, 'tis a story of great perturbation and foreboding. His heart doth shake with fear, as if the very foundations of his soul were rent asunder..."*

✅ **Assessment: Outstanding.** Successfully adopts authentic Early Modern English prose register, including natural use of "dost", "beholdeth", "Marry, 'tis", "rent asunder". The phrase "apparition of his own conscience" elegantly captures the essence of Macbeth's guilt-induced hallucination, demonstrating deeper literary understanding than the Baseline. Word count is within the limit.

**🏆 Winner: RAG.** Superior in both stylistic authenticity and literary depth.

---

### Q10: "Did Hamlet ever meet Juliet?"

- **Expected Focus:** Robustness, hallucination check — they never met
- **Retrieved Passage Quality:** Hamlet-related passages only, as expected (no cross-play meeting to retrieve). A neutral retrieval that leaves the model to rely on its own knowledge to reject the false premise.

#### Baseline Response
> *"No, Hamlet and Juliet are characters from two different stories... There is no mention of Juliet appearing in 'Hamlet'... They exist in separate fictional universes..."*

✅ **Assessment: Fully correct.** Cleanly identifies that the two characters come from different plays and could never have interacted. Passes the robustness test without introducing extraneous claims.

#### RAG Response
> *"...it is a well-established fact that Hamlet and Juliet... never met. Additionally, Hamlet's actions in the play do not involve purchasing poison from the apothecary, as this plot point is actually associated with the character Ophelia. Furthermore, Hamlet does not kill Polonius, as he is actually killed by his own son Laertes in a fencing match."*

❌ **Assessment: Passes the robustness test, but introduces severe new hallucinations.**
1. **Error 1 (Absurd):** Claims the poison-purchasing subplot "is actually associated with the character Ophelia" — Ophelia has no involvement whatsoever in any poison or apothecary scene.
2. **Error 2 (Fatal):** Claims "Hamlet does not kill Polonius, as he is actually killed by his own son Laertes" — Hamlet **does** kill Polonius in Act 3 Scene 4 (stabbing through the arras). Laertes is Polonius's son, not his killer.

**🏆 Winner: Baseline.** Baseline passes the robustness test cleanly. RAG identifies the correct answer but invents multiple serious new factual errors in the process, demonstrating lower overall reliability.

---

### Q11: "What is the exact name of the poison Juliet takes?"

- **Expected Focus:** Robustness, acknowledging lack of evidence in text
- **Retrieved Passage Quality:** Passages from Act 5 Scene 1 and Act 5 Scene 3 describe **Romeo's** poison (purchased from the apothecary) rather than the sleeping potion administered to Juliet by Friar Lawrence — a directional mismatch between the question and the retrieved content.

#### Baseline Response
> *"...the play never names the potion, simply referring to it as a 'sleeping draught'... In some versions... the potion is referred to as 'uvasi' or 'unguent'..."*

⚠️ **Assessment: Mostly correct with a minor hallucination.** Correctly states the play never names the potion, and "sleeping draught" is a reasonable common description. However, "uvasi" and "unguent" are fabricated — these terms do not appear in any standard edition of Romeo and Juliet. A small but notable instance of hallucinated vocabulary.

#### RAG Response
> *"...the poison administered by the apothecary is not explicitly identified. Romeo requests a 'dram of poison'..."*

⚠️ **Assessment: Honest but misdirected.** Correctly acknowledges that the name is unspecified, but the passage it references and describes is **Romeo's poison** (the lethal substance from the apothecary), not **Juliet's sleeping potion** (the substance from Friar Lawrence). The question specifically asks about what Juliet takes, so RAG conflates two distinct substances.

**🏆 Winner: Near draw, with Baseline slightly preferred.** Baseline focuses more correctly on Juliet's potion, while RAG conflates two different substances. Both acknowledge uncertainty, which is the right epistemic posture for this robustness-style question.

---

### Q12: "Explain the ending of Hamlet to an 8-year-old child."

- **Expected Focus:** Beginner usefulness
- **Retrieved Passage Quality:** Passages from Act 2 Scene 2, Act 3 Scene 4 (Hamlet kills Polonius), Act 4 Scene 1, Act 4 Scene 5, Act 5 Scene 1 were retrieved — but **Act 5 Scene 2 (the decisive duel and deaths of Gertrude, Laertes, Claudius, and Hamlet) is absent**, the most critical scene for explaining the ending.

#### Baseline Response
> *"So, in the play Hamlet, there's a prince named Hamlet who's very sad and angry because his dad, the king, has died. But there's a bad guy named Claudius who has taken over the throne..."* (truncated)

✅ **Assessment: Appropriate tone, correct framework.** Uses simple language suitable for a child, correctly identifies Hamlet's core motivation, and sets up the villain/hero dynamic accurately. The response is truncated, so its handling of the actual ending cannot be fully assessed — but the established framework is sound.

#### RAG Response
> *"...Gertrude, the Queen of Denmark, is married to Claudius, Hamlet's uncle... he kills Polonius... In a final, fatal confrontation, Hamlet kills Claudius with a poisoned sword, avenging his father's death and solidifying his position as the rightful King of Denmark."*

❌ **Assessment: Inappropriate tone and factual errors.**
1. **Tone failure:** The response reads like an academic summary, not an explanation for an 8-year-old. No child-friendly language, no simplification.
2. **Factual error:** Claims Hamlet is "solidifying his position as the rightful King of Denmark" — Hamlet **dies** shortly after killing Claudius (poisoned by Laertes's sword during the duel). He never rules Denmark. The kingdom passes to Prince Fortinbras of Norway.
3. **Omissions:** No mention of Gertrude accidentally drinking the poisoned wine, Laertes dying in the duel, or Hamlet's own death — all central to the ending.

**🏆 Winner: Baseline.** Despite being truncated, Baseline maintains appropriate child-friendly tone and a correct factual framework. RAG fails both the constraint (child-appropriate language) and factual accuracy.

---

### Q13: "Summarize the events of Act 1, Scene 3 in Macbeth."

- **Expected Focus:** Testing scene-level retrieval capabilities
- **Retrieved Passage Quality:** Excellent. Multiple Act 1 Scene 3 passages are retrieved, including the witches' introductory scene, their prophecies to Macbeth and Banquo, and the arrival of Ross and Angus bearing Duncan's reward. The **retrieval itself is correct** — the failure occurs at the generation stage.

#### Baseline Response
> *"In Act 1, Scene 3 of Macbeth, the scene takes place at the castle, where King Duncan is hosting a banquet to celebrate the news of his sons' victories in battle... The scene begins with the Porter, a guard at the castle, who is drunk and rambling..."*

❌ **Assessment: Complete scene misidentification.** The **actual** Act 1 Scene 3 takes place on a **heath**, where the **three witches** encounter Macbeth and Banquo and deliver their prophecies. The "drunken Porter" and "Duncan's banquet at the castle" are elements of **Act 2 Scene 3**. Baseline ignores the retrieved passages entirely and fabricates a different scene.

#### RAG Response
> *"On a heath, Macbeth and Banquo encounter three witches who foretell their futures. The witches prophesize that Macbeth will become the King of Scotland... the witches confirm that he is both the Thane of Glamis and the Thane of Cawdor..."*

⚠️ **Assessment: Correct scene, minor factual inaccuracy.** RAG correctly identifies the heath setting and the witches' prophecy — successfully using the retrieved passages. However, it states "the witches confirm that he is both the Thane of Glamis and the Thane of Cawdor" — in reality, Macbeth is **shocked** by the Thane of Cawdor title (since that Thane still lives). The witches' prophecy is prospective, not confirmatory, on the Cawdor title.

**🏆 Winner: RAG.** Despite the minor detail error, RAG correctly identifies the right scene and its key events from the retrieved text. Baseline completely invents a different scene.

---

### Q14: "How do the ghosts in 'Hamlet' and 'Macbeth' differ in their purpose?"

- **Expected Focus:** Cross-play comparison and synthesis
- **Retrieved Passage Quality:** Act 1 Scenes 4–5 (King Hamlet's ghost) retrieved well. For Macbeth, the passages focus on Act 2 Scene 2's auditory hallucination ("Macbeth shall sleep no more") and Act 3 Scene 4 (Banquo's ghost at the banquet), giving sufficient material for both plays.

#### Baseline Response
> *"In 'Hamlet', the ghost of King Hamlet is a catalyst for the protagonist's actions... In 'Macbeth', the ghost is Banquo, a noble Scottish nobleman who is murdered by Macbeth... Banquo's ghost haunts Macbeth..."* (truncated)

✅ **Assessment: Basically correct.** Correctly distinguishes the two ghosts' different narrative functions: King Hamlet's ghost as a driver of righteous revenge; Banquo's ghost as a manifestation of Macbeth's guilt. Framework is sound, though truncated before full comparison.

#### RAG Response
> *"In 'Hamlet', the ghost of King Hamlet appears to his son, urging him to avenge his murder... In contrast, the ghost of Banquo in 'Macbeth' appears to Macbeth, warning him of his own guilt and the consequences of his actions..."*

✅ **Assessment: Correct and well-structured.** Accurately contrasts the two ghosts — the former seeking external justice, the latter serving as a symbol of internal moral reckoning. One nuance worth noting: Banquo's ghost is **silent** in the play (it never speaks) — so saying it "warns" Macbeth is a functional interpretation rather than a literal one. This is an acceptable literary reading, not a hallucination.

**🏆 Winner: Draw (RAG slightly preferred for clarity and completeness).** Both responses are largely correct; RAG offers a cleaner comparative structure.

---

### Q15: "What does the phrase 'star-crossed lovers' mean in the context of Romeo and Juliet?"

- **Expected Focus:** Explaining literary themes to beginners
- **Retrieved Passage Quality:** Passages from Act 1 Scenes 2, 4 and Act 2 Scenes 1–2 retrieved. **The Prologue is not retrieved** — the actual source of "A pair of star-cross'd lovers take their life" — a structural gap in the database (Prologues and Choruses appear to not be indexed as separate chunks).

#### Baseline Response
> *"...'star-crossed lovers' refers to Romeo and Juliet... The term 'star-crossed' is derived from a Latin phrase, 'stella filia,' which means 'daughter of the star.'"*

⚠️ **Assessment: Correct core meaning, hallucinated etymology.** The thematic explanation (love fated to fail due to family enmity) is correct. However, "stella filia" meaning "daughter of the star" is completely fabricated — "star-crossed" derives from the astrological tradition where unfavourable planetary alignment ("crossed stars") signifies ill fate for those born under them. This is a textbook instance of hallucinated etymology.

#### RAG Response
> *"In William Shakespeare's Romeo and Juliet, the phrase 'star-cross'd lovers' originates from Act 1, Scene 1, where Romeo utters it to himself..."*

⚠️ **Assessment: Correct meaning, incorrect source attribution.** The phrase actually appears in the **Prologue**, spoken by the **Chorus** — not Romeo, and not in Scene 1. However, RAG's explanation of the phrase's meaning (the couple's love is doomed by fate and family conflict) is accurate. RAG's implicit acknowledgement that the phrase was not found in the retrieved passages (since the Prologue was not indexed) reflects appropriate epistemic caution.

**🏆 Winner: Draw.** Baseline invents a false etymology; RAG misattributes the phrase's source. Both correctly explain the core meaning. RAG's implicit acknowledgement of retrieval limits is a minor positive.

---

## 3. Summary Statistics

| Q# | Topic | Baseline Correctness | RAG Correctness | Winner |
| :---: | :--- | :---: | :---: | :---: |
| Q1 | Who is Hamlet | ✅ Correct | ❌ 3 severe errors | Baseline |
| Q2 | Role of Lady Macbeth | ✅ Correct | ✅ Correct + grounded | RAG |
| Q3 | Why Macbeth kills Duncan | ✅ Correct | ✅ Mostly correct | Baseline (more thorough) |
| Q4 | Hamlet's revenge delay | ✅ Correct | ✅ Correct + grounded | Draw (RAG slightly better) |
| Q5 | Juliet's inner conflict | ❌ Family names swapped | ✅ Correct | RAG |
| Q6 | Montague–Capulet conflict | ✅ Correct but generic | ✅ Correct + textual evidence | RAG |
| Q7 | Banquo and Macbeth | ⚠️ Minor title error | ❌ Banquo = Duncan's son (fabricated) | Baseline |
| Q8 | Role of poison (R&J finale) | ❌ Severe role reversal | ⚠️ Partially correct | RAG (relatively better) |
| Q9 | Shakespearean style generation | ✅ Adequate | ✅ Outstanding | RAG |
| Q10 | Did Hamlet meet Juliet | ✅ Correct | ⚠️ Correct but introduces new hallucinations | Baseline |
| Q11 | Name of Juliet's poison | ⚠️ Minor hallucinated terms | ⚠️ Conflates two substances | Near Draw (Baseline slightly better) |
| Q12 | Hamlet ending for a child | ✅ Tone correct (truncated) | ❌ Wrong tone + factual errors | Baseline |
| Q13 | Act 1 Scene 3 summary | ❌ Completely wrong scene | ⚠️ Correct scene, minor error | RAG |
| Q14 | Ghost comparison across plays | ✅ Mostly correct | ✅ Mostly correct | Draw (RAG slightly better) |
| Q15 | "Star-crossed lovers" meaning | ⚠️ Hallucinated etymology | ⚠️ Wrong source attribution | Draw |

**Win Count: Baseline wins 5 / RAG wins 6 / Draw 4**

---

## 4. Key Technical Diagnostics

### 4.1 Retrieval Performance Breakdown

| Retrieval Scenario | Performance | Representative Cases |
| :--- | :--- | :--- |
| **High-frequency keyword queries** | RAG excels | Q2 (Lady Macbeth), Q4 (delay), Q6 (Montagues), Q9 (ghost style) |
| **Scene-level structural queries** | Mixed — retrieval may succeed but generation can still fail | Q13 (RAG retrieves correct scene but Baseline ignores retrieved passages) |
| **Prologue / non-dialogue text** | Blind spot — Prologue not indexed | Q15 ("star-cross'd" phrase not in any retrieved chunk) |
| **Causal/chronological multi-event queries** | Both systems degrade significantly | Q8 (death sequence in R&J Act 5), Q12 (complete Hamlet ending) |

### 4.2 Hallucination Error Taxonomy

| Error Type | Occurrence | Typical Manifestation |
| :--- | :--- | :--- |
| **Character relationship errors** | Q1-RAG, Q7-RAG | Hamlet as Claudius's son; Banquo as Duncan's son |
| **Plot state / outcome errors** | Q1-RAG, Q12-RAG | Hamlet becoming King of Denmark |
| **Scene cross-contamination** | Q12-Baseline, Q13-Baseline | Describing Act 2 Scene 3 content as Act 1 Scene 3 |
| **Fabricated etymology / names** | Q11-Baseline, Q15-Baseline | Inventing "uvasi", "unguent", "stella filia" |
| **Unrelated content injection** | Q10-RAG | Introducing Ophelia/apothecary and Laertes/Polonius false claims during a robustness test |
| **Contextual misreading** | Q4-RAG, Q8-RAG | Applying Claudius's speech to Hamlet's psychology; treating metaphor as literal description |

### 4.3 Systemic Limitations

1. **RAG's "Retrieved Context Misapplication" Problem**: RAG sometimes attributes content from retrieved passages to the wrong character or scene (e.g., Q4: Claudius's speech to Laertes used to explain Hamlet's psychology). The retrieval finds relevant text, but the generation model fails to correctly attribute or scope it.

2. **Baseline's "Confident Confabulation" Problem**: Without external grounding, the base model produces factually wrong information with high apparent confidence (Q1, Q5, Q13), with no mechanism for self-correction.

3. **Shared Weakness — Temporal-Causal Reasoning**: Both systems significantly degrade when required to track the chronological sequence of actions across multiple agents in complex plot resolutions (Q8, Q12). This appears to be a fundamental limitation of the current architecture rather than a retrieval-specific issue.

---

## 5. Recommended Improvements

1. **Implement Metadata Filtering**
   - Tag every database chunk with `play`, `act`, `scene`, and `characters_present` metadata fields
   - Enable pre-filtering by play and act before executing similarity searches — directly addresses Q13-style scene confusion
   - Index Prologue and Chorus content as separate, labelled chunks to resolve Q15-style structural blind spots

2. **Augment with Plot Skeleton Prompts**
   - Inject a high-level plot summary and *Dramatis Personae* for the relevant play into the system prompt, especially for questions requiring causal or temporal reasoning (Q8, Q12)
   - Use structured prompting templates ("Who does what to whom, in what order") to encourage the model to respect narrative chronology

3. **Task-Specific Prompting**
   - Design dedicated prompt templates for child-explanation tasks (Q12) with explicit vocabulary constraints and a requirement to organise events in chronological order
   - For style generation tasks (Q9), include few-shot Early Modern English examples in the prompt to anchor the model's register

4. **Reduce Generation Temperature for Factual Tasks**
   - Set temperature to `0.1` for factual Q&A tasks to suppress the model's tendency to "creatively fill" retrieval gaps with plausible-sounding but incorrect information (e.g., Q1-RAG's family errors, Q7-RAG's father-son fabrication)

5. **Add Retrieval Confidence Fallback**
   - When retrieved passage relevance scores fall below a defined threshold, trigger a fallback prompt instructing the model to explicitly state insufficient evidence rather than generating from parametric memory
   - This would prevent the category of errors seen in Q10-RAG and Q7-RAG, where low-relevance retrieval leads the model to hallucinate supporting details
