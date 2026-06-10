# SIGNAL-Preop Questionnaire Structure

SIGNAL-Preop is a two-stage, structured preoperative psychiatric screening interview. It first screens broad symptom domains and then activates domain-specific follow-up modules when a core symptom is endorsed. The instrument is organized around depressive symptoms, generalized anxiety symptoms, insomnia-related symptoms, and general clinical/contextual factors.

## 1. Overall branching logic

### Stage 1: Core screening layer

Participants first answer brief screening items covering:

* Depressed mood
* Loss of interest or anhedonia
* Excessive worry or anxiety
* Sleep disturbance
* Functional impact
* General clinical context and risk-related factors

If a participant does not endorse a core domain, downstream items in that module are bypassed and encoded as a negative response for modeling.

If a participant endorses a core domain, the corresponding follow-up module is activated.

### Stage 2: Domain-specific modules

Activated modules collect additional information on symptom characteristics, duration, frequency, severity, functional impact, and related clinical features.

The final output is an ordered sequence of structured question–answer pairs, preserving both standardized prompts and patient-generated responses.

---

## 2. Module structure

## A. Depressive symptom module

Triggered by endorsement of depressed mood, loss of interest, or related depressive symptoms.

| Code  | Item concept                          | Purpose                                      |
| ----- | ------------------------------------- | -------------------------------------------- |
| DM    | Depressed mood                        | Screens persistent low mood or sadness       |
| LOIOA | Loss of interest or anhedonia         | Screens reduced interest or pleasure         |
| FORA  | Fatigue or reduced activity           | Assesses reduced energy or activity          |
| DAOWL | Decreased appetite or weight loss     | Assesses appetite or weight change           |
| CIAOW | Changes in appetite or weight         | Captures broader appetite or weight changes  |
| EG    | Excessive guilt                       | Screens inappropriate or excessive guilt     |
| FOW   | Feelings of worthlessness             | Screens negative self-evaluation             |
| IC    | Impaired concentration                | Assesses concentration difficulties          |
| PR    | Psychomotor retardation               | Assesses slowed movement or speech           |
| HYP   | Hypersomnia                           | Assesses excessive sleep duration            |
| ROLL  | Reduced or lost libido                | Assesses reduced sexual interest             |
| STROC | Sensitivity to rejection or criticism | Assesses interpersonal sensitivity           |
| SW    | Social withdrawal                     | Assesses reduced social engagement           |
| BD    | Behavioral disorganization            | Assesses behavioral or functional disruption |

---

## B. Generalized anxiety symptom module

Triggered by endorsement of excessive worry, uncontrollable worry, or anxiety-related distress.

| Code  | Item concept                         | Purpose                                         |
| ----- | ------------------------------------ | ----------------------------------------------- |
| EWAMA | Excessive worry about multiple areas | Screens broad and excessive worry               |
| WC    | Worry content                        | Identifies main worry themes                    |
| UW    | Uncontrollable worry                 | Assesses difficulty stopping worry              |
| DOW   | Duration of worry                    | Assesses persistence of worry                   |
| DCW   | Difficulty controlling worry         | Captures impaired control over worry            |
| DC    | Difficulty concentrating             | Assesses anxiety-related concentration problems |
| POCR  | Physical or cognitive restlessness   | Assesses restlessness or tension                |
| FRTA  | Fatigue related to anxiety           | Assesses anxiety-related fatigue                |
| GS    | Globus sensation                     | Captures somatic anxiety symptoms               |
| SDDTW | Sleep disturbance due to worry       | Links worry to sleep disturbance                |

---

## C. Insomnia symptom module

Triggered by endorsement of sleep disturbance or sleep-related impairment.

| Code | Item concept                   | Purpose                                 |
| ---- | ------------------------------ | --------------------------------------- |
| SD   | Sleep disturbances             | Screens general sleep problems          |
| DSD  | Daily sleep duration           | Assesses average sleep time             |
| SOD  | Sleep onset difficulty         | Assesses difficulty falling asleep      |
| MA   | Middle-of-night awakenings     | Assesses nocturnal awakenings           |
| EMA  | Early morning awakenings       | Assesses early waking                   |
| FN   | Frequent nightmares            | Assesses nightmare frequency            |
| NS   | Non-restorative sleep          | Assesses unrefreshing sleep             |
| FOSD | Frequency of sleep disturbance | Assesses how often sleep problems occur |

---

## D. General clinical and contextual module

Collected to support interpretation of psychiatric symptoms and clinical triage.

| Code | Item concept         | Purpose                                    |
| ---- | -------------------- | ------------------------------------------ |
| IODL | Impact on daily life | Assesses functional impairment             |
| OAR  | Onset and recurrence | Assesses timing and recurrence of symptoms |
| PI   | Physical illness     | Captures relevant physical disease context |
| SU   | Substance use        | Screens substance-related factors          |

---

## 3. Encoding for modeling

For each participant, the interview produces an ordered sequence of question–answer pairs.

* Completed items retain the participant’s original response.
* Bypassed modules are encoded using the corresponding negative screening response.
* Question and answer segments are separated using fixed delimiters.
* The resulting sequence is used as input for language modeling.

Example structure:

```text
[Q] Core depressed mood screening
[A] Negative

[Q] Core worry screening
[A] Positive

[Q] Worry content follow-up
[A] Patient-generated response

[Q] Difficulty controlling worry
[A] Patient-generated response
```

---

## 4. Notes

This document describes the questionnaire structure and branching logic at a conceptual level. It does not reproduce the exact clinical wording of SIGNAL-Preop items. Local implementations should adapt item wording, safety escalation procedures, and clinical review pathways according to institutional requirements.
