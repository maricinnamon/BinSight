# App Store metadata — English and Ukrainian

Ready to paste into App Store Connect. Character counts are against Apple's
limits; none exceed them.

---

## English (en-GB / en-US)

**App name** (30 max) — `BinSight` *(8)*

**Subtitle** (30 max) — `On-device waste detection` *(25)*

**Promotional text** (170 max, editable without review)

> Point your camera at paper, plastic or metal and see it recognised instantly.
> Everything runs on your iPhone — no account, no upload, no internet needed. *(148)*

**Keywords** (100 max, comma-separated, no spaces)

```
recycle,recycling,waste,sorting,plastic,paper,metal,scanner,camera,offline,ai,detect,eco,bin
```
*(93)*

**Description**

> BinSight points your camera at a piece of waste and tells you what it is made
> of: paper, plastic or metal. It draws a box around what it finds and shows how
> confident it is.
>
> Everything happens on your iPhone. There is no account, no sign-in, and no
> internet connection required. Camera frames are never stored and never
> uploaded — the recognition model runs entirely on the device, using Apple's
> Neural Engine where available.
>
> WHAT IT DOES
> • Recognises paper, plastic and metal from the live camera
> • Draws a bounding box around the object it is most confident about
> • Shows the confidence as a percentage, so you can judge it yourself
> • Offers brief, generic handling advice for each material
> • Works completely offline, in aeroplane mode, anywhere
>
> PRIVACY BY CONSTRUCTION
> BinSight contains no networking code at all. Not "we promise not to upload" —
> there is no code that could. No analytics, no advertising, no third-party
> SDKs, no crash reporting. The only thing it stores is your language choice.
>
> HONEST ABOUT ITS LIMITS
> BinSight recognises three materials, not every kind of waste. It is right
> about roughly three quarters of the boxes it draws, and it misses things —
> especially crumpled paper and transparent plastic. It shows you its confidence
> so you can decide whether to trust it.
>
> The advice it gives is generic material guidance. Local recycling rules differ
> enormously between councils and countries, and BinSight does not know yours.
> Always follow your local scheme.
>
> LANGUAGES
> English and Ukrainian, switchable inside the app.

---

## Українська (uk)

**Назва застосунку** (30 max) — `BinSight` *(8)*

**Підзаголовок** (30 max) — `Розпізнавання на пристрої` *(25)*

**Рекламний текст** (170 max)

> Наведіть камеру на папір, пластик або метал — і побачите результат одразу.
> Усе працює на вашому iPhone: без акаунта, без завантажень, без інтернету. *(146)*

**Ключові слова** (100 max)

```
сортування,переробка,відходи,пластик,папір,метал,сміття,камера,офлайн,розпізнавання,еко,штучний
```
*(99)*

**Опис**

> BinSight наводить камеру на предмет і каже, з чого він: папір, пластик чи
> метал. Обводить знайдене рамкою і показує, наскільки він у цьому впевнений.
>
> Усе відбувається на вашому iPhone. Не потрібен акаунт, вхід чи інтернет. Кадри
> ніколи не зберігаються й не надсилаються — модель розпізнавання працює цілком
> на пристрої, за потреби задіюючи Neural Engine.
>
> ЩО ВІН УМІЄ
> • Розпізнає папір, пластик і метал із живої камери
> • Обводить рамкою той предмет, у якому впевнений найбільше
> • Показує впевненість у відсотках, щоб ви могли оцінити самі
> • Дає коротку загальну пораду щодо кожного матеріалу
> • Працює повністю офлайн — навіть у режимі польоту
>
> ПРИВАТНІСТЬ ЗА БУДОВОЮ
> У BinSight немає мережевого коду взагалі. Не «ми обіцяємо нічого не
> надсилати», а просто немає коду, який міг би це зробити. Жодної аналітики,
> реклами, сторонніх SDK чи збору збоїв. Єдине, що зберігається, — обрана мова.
>
> ЧЕСНО ПРО ОБМЕЖЕННЯ
> BinSight розпізнає три матеріали, а не всі види відходів. Приблизно три чверті
> намальованих рамок правильні, і він дечого не помічає — особливо зім'ятого
> паперу й прозорого пластику. Він показує свою впевненість, щоб ви вирішували
> самі.
>
> Поради загальні, щодо матеріалу. Правила переробки дуже різняться між
> громадами й країнами, і BinSight не знає ваших. Завжди дотримуйтесь місцевих.
>
> МОВИ
> Англійська й українська, перемикаються всередині застосунку.

---

## Support and marketing URLs

Both are **required** by App Store Connect and neither exists yet:

- Support URL — the GitHub repository's issues page is acceptable for a first release
- Marketing URL — optional, may be left blank
- Privacy Policy URL — **required**. `docs/app-store/PRIVACY.md` must be published
  somewhere reachable over HTTPS (GitHub Pages or the raw file view both work).

## Category

- Primary: **Utilities**
- Secondary: **Education** *(optional; the app teaches material recognition)*

Not Lifestyle — the app is a tool, and Utilities is where a camera scanner is
looked for.
