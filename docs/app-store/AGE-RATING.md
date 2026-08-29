# Age rating questionnaire

Expected outcome: **4+** (Apple), **PEGI 3** equivalent.

Every answer below is **None / No**, and each is a statement about the code
rather than an intention.

| Question | Answer | Why |
|---|---|---|
| Cartoon or Fantasy Violence | None | No violence of any kind |
| Realistic Violence | None | |
| Prolonged Graphic or Sadistic Realistic Violence | None | |
| Profanity or Crude Humor | None | All copy is functional; both languages reviewed |
| Mature/Suggestive Themes | None | |
| Horror/Fear Themes | None | |
| Medical/Treatment Information | None | Material handling advice is not medical |
| Alcohol, Tobacco, or Drug Use or References | None | The model has no such classes and the copy never mentions them |
| Simulated Gambling | None | |
| Sexual Content or Nudity | None | |
| Graphic Sexual Content and Nudity | None | |
| Contests | None | |
| Unrestricted Web Access | **No** | There is no web view and no networking code at all |
| Gambling and Contests | None | |

## Additional declarations

| Question | Answer |
|---|---|
| Does the app contain, display, or access third-party content? | **No** |
| Does the app use encryption? | **No** — no networking, no custom crypto. Exempt from export compliance; set `ITSAppUsesNonExemptEncryption = NO` |
| Does the app include advertising? | **No** |
| Is the app made for kids? | **No** — not submitted to the Kids Category, though the content is suitable |
| Does the app collect data? | **No** — see `DATA-COLLECTION.md` |

## One thing to think about before submitting

BinSight points a camera at the world and a person will inevitably point it at
another person. The app does not detect, recognise or store people — it has three
classes and none of them is a face — and no frame is retained. There is no
face-detection API in the binary. This is worth stating in review notes only if a
reviewer raises it.
