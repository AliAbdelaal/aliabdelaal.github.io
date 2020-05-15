---
title: "مقدمة في علم الآلة"
tagline: "في هذا المقال البسيط سوف نتعرض بشئ من التفصيل عن ماهية تعليم الآلة ما المقصود بها تحديدا و كيف تتعلم الآلة مع التطرق لبعض التفاصيل الرياضية لهذه العملية"
excerpt: "في هذا المقال البسيط سوف نتعرض بشئ من التفصيل عن ماهية تعليم الآلة ما المقصود بها تحديدا و كيف تتعلم الآلة مع التطرق لبعض التفاصيل الرياضية لهذه العملية"
header:
  overlay_image: https://images.unsplash.com/photo-1527430253228-e93688616381?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1491&q=80
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
  teaser: https://images.unsplash.com/photo-1527430253228-e93688616381?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1491&q=80
categories:
  - Blog
tags:
  - machine learning
  - arabic
toc: true
toc_sticky: true
---
<div dir="rtl">
مجال تعلم الآلة او (machine learning) هو احد فروع الذكاء الاصطناعي و يشتمل بداخله علم التعلم العميق (deep learning) 
</div>

![ai-vs-ml]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-ml/ai-and-ml.png){: .align-center}

<div dir="rtl">
و عرفه العالم آرثر سامويل 
</div>

{:refdef: .notice--info}
<div dir="rtl">
ان تعطي اجهزة الكمبيوتر القدرة على التعلم بدون ان تتم برمجتها بشكل صريح

<cite><a href="https://www.wikiwand.com/simple/Machine_learning">المصدر</a></cite>
</div>
{:refdef}

<div dir="rtl">
لتوضيح هذا العنوان بشكل افضل، لنستعرض الفرق بين البرمجة الحرفية و تعليم الآلة بشكل مبسط
</div>

## البرمجة التقليدية (الحرفية)
{: .text-right}

<div dir="rtl">
لنقل على سبيل المثال اننا نود ان نميز بين اذا ما كان في الصورة قطة ام لا

<center><img src='https://images.unsplash.com/photo-1520315342629-6ea920342047?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80'></center>
<center><a href="https://images.unsplash.com/photo-1520315342629-6ea920342047?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80">المصدر</a></center>

ففي حالة البرمجة التقليدية سوف نقوم بوضع جميع القواعد التي تحتاجها الآلة حتى تميز هذا القطة مثل ان اذنها يجب ان تكون مثلثة الشكل قليلا و ان وجهها يجب ان يكون دائريا نوعا ما و ايضا ان تحتوى على فراء، وهكذا إلى اخر القواعد التي يمكنك كتابتها

وفي حالة ان احد هذه القواعد لم تماثل ماقمت بتعريفه لن نتمكن من ان نتعرف على القطة في الصورة للاسف.
</div>

```python
if ear is triangle:
  if face is circular:
    if object has furr:
      if image is cute:
          "it's a cat!"
```

![rule-based]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-ml/rule-based.png){: .align-center}

<div dir='rtl'>
بالطبع يمكنك الملاحظة ان هنا لن نتمكن من تغطية جميع المواصفات التي تميز القط، ناهيك عن اختلاف القرب او البعد عن الصورة على سبيل المثال او تغيير مكان القطة في الصورة او حتى طريقة تحديد هذه المواصفات

بالتالي هذه ليست الطريقة الافضل في هذه المهمة، لذا تعالى معي نستعرض الطريقة الاخرى و هي باستخدام تعليم الآلة
</div>

## طريقة تعليم الآلة (التعلم من البيانات)
{: .text-right}

<div dir='rtl'>
في هذه الطريقة نعتمد بالاساس على ان نوفر صور مختلفة لقطط و اي شئ غير القطط و نترك الامر لما سوف ندعوه <b>نموذج</b> كي يتعلم ماهية القط و كيف يمكنه تمييزه عن غيرها من الاشياء.

بالطبع هذه الطريقة تعتمد على ان يتوفر صور عدة للقطط وكذلك لغير القطط حتى يتمكن نوذجنا من معرفة القط من غيره و من هذا المنطلق تنشأ الحاجة الى كثير من البيانات حتى نتمكن من عمل نوذج قوي يمكنه التمييز، و توفر البيانات بكثرة في العقد الاخير سمح بتقدم هائل في مجال تعلم الآلة والذكاء الصناعي بشكل عام.
</div>

```python
cats = [cat1, cat2, cat3, ...]
machine_learning_model.learn_from(cats)
machine_learning_model.predict(new_image)
```

![ml-based]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-ml/ml-based.png){: .align-center}

<div dir='rtl'>
حسنا عرفنا الآن الفرق الاساسي بين التعلم من البيانات و الطرق المعتمدة على البيانات التي تطيق عليها (data driven approaches) وكذلك الطرق التي تعتمد على ان نعطيها القواعد التي تم تصميمها يدويا (rule based approaches).

ولكن كيف للآلة ان تتعلم هكذا صفات؟ و ان تفهم النمط المتواجد في البيانات و بناءا عليه تمييز البيانات الجديدة التي لم يراها من قبل؟

لنفهم هذا دعنا ننظر الى مثال بسيط في هذا الشأن
</div>

## استخراج النمط من البيانات
{: .text-right}

<div dir='rtl'>
لنقل على سبيل المثال ان لديك منزل تريد ان تبيعه، ولا تعرف اي سعر قد يكون مناسب لك و لكنك تعلم ان اخر 3 بيوت تم بيعهم في المنطقة المحيطة لك كانت اسعارهم و مواصفاتهم كالاتى
</div>

<table>
  <tbody>
    <tr>
      <td align="center">الطابق الواقع فيه المنزل<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">مساحة المنزل<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">سعر البيع<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
    </tr>
    <tr>
        <td align="center">الثالث<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">300<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">9300<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
    </tr>
    <tr>
      <td align="center">الاول<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">250<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">7600<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
    </tr>
        <tr>
      <td align="center">الخامس<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">100<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">800<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
    </tr>

  </tbody>
</table>

<div dir='rtl'>
و لديك مواصفات منزلك و الذي يقع في <b>الطابق الثاني</b> و مساحته <b>220</b> اذا ما هو السعر المناسب لمنزلك؟

بالطبع يمكننا ان نقوم بتحويل هذه المسألة إلى معادلات في متغيريين كما يلي

اذا اعتبرنا ان الطابق هو متغير X و ان المساحة هي متغير Y اذا يمكننا ان نصف اول منزلين كما يلي 
</div>

$$ 3X + 300Y = 9300 $$

$$ 1X + 250Y = 7600 $$

<div dir='rtl'>
نستطيع حل هاتان المعادلتان سويا و استخراج X بدلالة Y ثم ايجاد قيمتهما سويا كما يلي
</div>

$$ \because X = 7600 - 250 Y $$

$$ \therefore 3 (7600 - 250 Y) + 300 Y = 9300 $$

$$ \therefore Y = 30 $$

$$ \therefore X = (7600 - 250 * 30) = 100 $$

<div dir='rtl'>
و هكذا يمكننا ان نحسب ثمن المنزل الذي نريد ان نبيعه كالتالي
</div>

$$ 100 * 2 + 30 * 220 = 6800 $$

<div dir='rtl'>
هنا يمكنك تفسير قيمة X و Y على انهم اهمية الطابق الذي تقع .فيه الشقة و مساحتها على الترتيب
<br>
<br>
بالتالي هنا نرى ان اهمية الطابق قيمتها 100 و اهمية المساحة قيمتها 30.
<br>
<br>
ما فعلناه للتو هو تخصيص ما يسمى <b>اوزان او(weights)</b> للمتغيرات او الخصائص التي لدينا وهي الطابق و المساحة، وهذا هو المبدأ الأساسي الذي تقوم عليه عملية التعلم، اذا دعنا نرى اشكال مختلفة لهذه المتغيرات او الخصائص و التي تقوم بتوصيف البيانات التي لدينا حتى نخصص لها اوزان و قيم لنستطيع ان نتعلم اهميتها.
</div>

## استخراج الخصائص من البيانات و توصيفها
{: .text-right}
