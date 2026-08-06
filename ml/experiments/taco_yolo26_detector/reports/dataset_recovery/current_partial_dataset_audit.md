# Current partial dataset audit

State of the local TACO copy **before** recovery.

| | |
|---|---:|
| Official image records | 1500 |
| Official annotation records | 4784 |
| Official categories | 60 |
| **Local image files** | **602** |
| **Missing image records** | **898** |
| Failed Flickr downloads | 898 |
| Mapped annotations, full dataset | 3144 |
| Mapped annotations, locally available | 1290 |
| **Mapped annotations lost to missing images** | **1854** |

Of the 898 missing images, **757** contain at least one annotation that maps to a BinSight class — the rest would not have contributed to training even if downloaded.

Download failures were `{'HTTP 429': 898}`: Flickr rate limiting, not dead links.

