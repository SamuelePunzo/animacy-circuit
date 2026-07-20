# Named Entity Grammar Report

## Target Counts

- Animate named entities: 110
- Inanimate/non-person named entities: 108
- Inanimate subtype counts: {'GPE_OR_LOCATION': 50, 'ORG': 30, 'PRODUCT_OR_WORK': 28}

## Surface Grammar Judgment

- `by the <named entity>` is generally ungrammatical for ordinary proper names.
- `by <named entity>` is the correct surface form for people, places, organizations, products, and works.
- Organizations such as `Google` are grammatically good passive agents in `by Google`.
- Locations such as `Paris` are grammatically well-formed in `by Paris`, but often require metonymic interpretation as a government, institution, or place-associated actor.

## Examples: Current Template

- The manuscript was annotated by the Michael Jackson
- The manuscript was annotated by the Taylor Swift
- The manuscript was annotated by the Beyonce
- The tunnel was secured by the Michael Jackson
- The tunnel was secured by the Taylor Swift
- The tunnel was secured by the Beyonce
- The database was archived by the Michael Jackson
- The database was archived by the Taylor Swift
- The database was archived by the Beyonce
- The manuscript was annotated by the Paris
- The manuscript was annotated by the London
- The manuscript was annotated by the Berlin
- The tunnel was secured by the Paris
- The tunnel was secured by the London
- The tunnel was secured by the Berlin
- The database was archived by the Paris
- The database was archived by the London
- The database was archived by the Berlin

## Examples: Named-Entity Template

- The manuscript was annotated by Michael Jackson
- The manuscript was annotated by Taylor Swift
- The manuscript was annotated by Beyonce
- The tunnel was secured by Michael Jackson
- The tunnel was secured by Taylor Swift
- The tunnel was secured by Beyonce
- The database was archived by Michael Jackson
- The database was archived by Taylor Swift
- The database was archived by Beyonce
- The manuscript was annotated by Paris
- The manuscript was annotated by London
- The manuscript was annotated by Berlin
- The tunnel was secured by Paris
- The tunnel was secured by London
- The tunnel was secured by Berlin
- The database was archived by Paris
- The database was archived by London
- The database was archived by Berlin

## Recommendation

Use the bare `by` template for named-entity completions and do not feed these lists into the current single-token common-noun target metric without a multi-token sequence-probability metric.
