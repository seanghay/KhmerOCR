## KhmerOCR

A Fast Khmer Optical Character Recognition (Document only).  The model was trained on 3M text lines with over 800+ Khmer fonts.

> ⚠️ NOTE: Only works with a single text line at the moment. Text detection will be added in the future.

### Install

```shell
pip install -e .
```

### Usage

```shell
python recognize.py image.jpg
```

**Input Image**

<img src=image.jpg >

**Output**

```
យេម៉ែនលង់ក្នុងសង្គ្រាមតាំងពីពួកឧទ្ទាមហ៊ូទីគាំទ្រដោយអ៊ីរ៉ង់ចូលគ្រប់គ្រងរដ្ឋធានីសាណានិងដែនដីមួយភាគធំទៀតនៅឆ្នាំ២០១៤។មួយឆ្នាំ
```

---

**Input Image**

<img src=image2.jpg >


**Output**

```
ក្រសួងមហាផ្ទៃឱ្យត្រៀមជម្លេរសពលរដ្ឋខ្មែរនៅថៃវិលមកស្រុកវិញបើមានការប៉ះទដ្គិចគ្នា
```

---

### Milestones

- [ ] Add English & Symbols
- [ ] Add Text Detection
- [ ] Add ONNXRuntime
- [ ] Add C/C++ Inference Engine

### License

```
MIT License

Copyright (c) 2026 Seanghay Yath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
