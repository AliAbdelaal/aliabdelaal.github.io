---
title: "Introduction to NLP in Arabic - part 2"
tagline: "نظرة عامة في مجال معالجة اللغة باستخدام خوارزميات تعلم الآلة الجزء الثاني"
excerpt: "نظرة عامة في مجال معالجة اللغة باستخدام خوارزميات تعلم الآلة الجزء الثاني"
header:
  overlay_image: https://unsplash.com/photos/LBKxu77KpSY/download?force=true
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
  teaser: https://unsplash.com/photos/LBKxu77KpSY/download?force=true
categories:
  - Blog
tags:
  - arabic
  - natural language processing
  - machine learning
toc: true
toc_sticky: true
toc_label: "المحتويات"
---

<script src="https://unpkg.com/vanilla-back-to-top@7.2.1/dist/vanilla-back-to-top.min.js"></script>
<script>addBackToTop({
  diameter: 56,
  backgroundColor: 'rgb(128, 128, 128)',
  textColor: '#fff'
})</script>

## مقدمة
{: .text-right}

<div dir='rtl'>
هذا المقال تكملة للمقال السابق، يمكنك الاطلاع عليه من <a href='{{ site.url }}{{ site.baseurl }}/blog/intro-to-nlp-p1/'>هنا</a>
</div>

<div dir='rtl'>
في المقال السابق تحدثنا عن نماذج مختلفة لمعالجة النصوص باستخدام خوارزميات تعلم الآلة، وتوقفنا عند بعض النقاط عند كل نموذج، دعنا نراجعها سريعا.
</div>

<div dir='rtl'>
<ul>

<li><div dir='rtl'>
ذكرنا في المرة السابقة استخدام Bag of words و وضحنا ان هذا النموذج لايأخذ في الحسبان ترتيب الكلمات وايضا لا يستطيع فهم معاني الكلمات فبالنسبة لهذا النموذج كل الكلمات سواء و الفارق هو عدد مرات تكرارهم.
</div></li>

<li><div dir='rtl'>
ذكرنا ايضا كيف ان استخدام معدل تكرار الكلمة في النصوص (TFIDF) يمكن ان يساعد في تميز الكلمات المهمة عن باقي الكلمات ولكنه ايضا يعاني من نفس مشاكل الطريقة السابقة.
</div></li>

<li><div dir='rtl'>
ثم ذكرنا نموذج متجهات الكلمات word2vec و ذكرنا كيف ان استخدام هذا النموذج يساعد في الحصول على متجه يعبر عن معنى الكلمات ولكن مازال لدينا بعض الملحوظات مثل ان الكلمة تحصل على نفس المتجه بغض النظر عن اختلاف السياق.
</div></li>

</ul>
</div>


<div dir='rtl'>
في هذا المقال سنتعرف على كيفية التغلب على المشاكل السابقة باستخدام نماذج اكثر تطورا، سنتحدث عن الجمع بين نموذج word2vec و نموذج CNN لجمع اكتر من متجه و محاكاة طريقة ال N-Grams. أيضا سنتحدث عن محاكاة البيانات التسلسلية (sequential data) باستخدام نموذج RNN.</div>

## كيف يعمل word2vec
{: .text-right}
 
<div dir='rtl'>
فهم كيفية عمل هذا النموذج سوف تساعدنا على معرفة حدوده بشكل افضل و ربما كيفية التغلب عليها!
</div>

<div dir='rtl'>
يوجد نوعان من نموذج word2vec هما CBOW (continous bag of word) و skip-gram، النوعان يختلفان في طريقة تدريبهما قليلا لكن الاستخدام لاحقا يكون في نفس السياق الا وهو توليد متجهات ذات معنى للكلمة.
</div>

<br>

![word2vec-arch]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/word2vec.png){: .align-center}
<center><a href="https://arxiv.org/abs/1301.3781">المصدر</a></center>

### CBOW
{: .text-right}

<div dir='rtl'>
طريقة ال CBOW تعتمد على تعلم الكلمة الوسطى من الكلمات المحيطة، هذا عن طريق ادخال متجهات الكلمات المحيطة و محاولة تعلم الكلمة الوسطى
</div>

<div dir='rtl'>
بالطبع في البداية لن يتوقع النموذج الكلمة الصحيحة ولكن مع التدريب و محاولة تقليص الخطأ باستخدام اسلوب <a href='{{ site.url }}{{ site.baseurl }}/blog/gradient-descent-family/'>النزول التدريجي في معدل الخطأ (gradient descent)</a> يتحسن اداء النموذج ليستطيع توقع الكلمة الصحيحة اذا ادخلنا الكلمات المحيطة، على سبيل المثال عندما ندخل للنموذج اريد ان ... كوب عصير، سيقوم النموذج بتوقع ان الكلمة الوسطى هنا هي "اشرب" لان هذا ما يدل عليه السياق.
</div>

<div dir='rtl' class='notice--info'>
نماذج التعلم العميق (deep learning) تعتبر بالاساس معادلات معقدة تحاول تخمين ما هي القيم التقريبية للمتغيرات التي تكون هذه المعادلات في البداية يتم تخمين قيم عشوائية لهذه المتغيرات و يتم احتساب قيم التوقعات و مقارنتها بالقيم الحقيقية و احتساب معدل الخطأ بينهما، و باستخدام خصائص التفاضل يمكننا معرفة التغييرات المطلوبة في كل متغير حتى نقوم بتقليل معدل الخطأ و يحدث هذا في عملية ال (gradient descent) يمكنك مراجعة تفاصيل اكثر في <a href='{{ site.url }}{{ site.baseurl }}/blog/gradient-descent-family/'>مقال سابق</a> حيث نتحدث عن تفاصيل هذه العملية بشكل اكبر.
</div>

<br>

![word2vec-cbow](https://cdn-images-1.medium.com/max/800/1*d66FyqIMWtDCtOuJ_GcqAg.png){: .align-center}
<center><a href='https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html'>المصدر</a></center>

### Skip-gram
{: .text-right}

<div dir='rtl'>
طريقة ال skip-gram تعتمد على تعلم السياق من الكلمة الوسطى (عكس الطريقة السابقة).
</div>

<div dir='rtl'>
لتبسيط طريقة تعلم هذا النموذج نقوم بتحضير المدخلات لهذا النموذج على النحو التالي.
</div>

<table>
  <thead>
    <tr>
      <td align="center">السياق<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">الكلمة قبل<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">الكلمة بعد<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
      <td align="center">صحة السياق<br>
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
    </thead>
    <tbody>
    <tr>
        <td align="center">شرب<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">اريد<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">العصير<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
      <td align="center">1<br>
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
      <td align="center">اكل<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">اريد<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>        
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;</span>
      </td>
      <td align="center">العصير<br>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>    
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>     
        <span>&nbsp;&nbsp;</span>        
      </td>
      <td align="center">0<br>
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
يمكنك ان ترى هنا اننا استخدمنا كلمة واحدة قبل و بعد كلمة السياق لكن يمكن ان يكون اكثر او اقل (اقل بان نستخدم كلمة واحدة قبل او بعد).
</div>

<div dir='rtl'>
هنا يصبح هدف النموذج ان يتوقع إذا كانت الكلمات المعطاه تنتمي الي نفس السياق ام لا و ايضا في البداية لن يتوقع النموذج الاجابة الصحيحة ولكن بعد التدريب و التصحيح يمكنه ذلك.
</div>

<br>
![word2vec-skip-gram](https://cdn-images-1.medium.com/max/800/1*4Uil1zWWF5-jlt-FnRJgAQ.png){: .align-center}
<center><a href='https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-skip-gram.html'>المصدر</a></center>

<div dir='rtl' class='notice--info'>
استخدام نماذج ايجابية و سلبية في التدريب يعد من اشهر الطرق في تدريب نماذج تعلم اللغة و تسمى هذه الطريقة بال (negative sampling) يمكنك الاطلاع على معلومات اكثر من خلال <a href='https://www.coursera.org/lecture/nlp-sequence-models/negative-sampling-Iwx0e'>هذا الفيديو القصير</a>.
</div>

### الاستخدام بعد التدريب
{: .text-right}

<div dir='rtl'>
بعد انتهاء عملية التدريب نقوم باستخراج مرحلة تحويل النص الي متجه (embedding layer) بعد ان تعلمت كيف تصف الكلمة المعطاه بمتجه يمثل معناها، و في هذه المرحلة كل كلمة تم التدريب عليها لها متجه ثابت الان لا يمكن تغييره بناءا على السياق الخاص بالجملة و لكنه بالتأكيد افضل من استخدام الطرق السابقة مثل (bag of words) للتعبير عن النصوص.
</div>

![word-embeddings](https://www.tensorflow.org/text/guide/images/embedding2.png){: .align-center}
<div dir='rtl'>
<center>مثال على استخدام ال embdding layer لمتجه حجمه 4 متغيرات. <a href='https://www.tensorflow.org/text/guide/word_embeddings'>المصدر</a></center></div>


## استخدام ال CNN في معالجة اللغة
{: .text-right}

<div dir='rtl'>
تعد ال CNN (Convolutional Neural Network) الاكثر شهرة و استخداما في مجال معالجة الصور ولكن ايضا تستخدم في معالجة النصوص كما سنرى في الامثلة الاتية.
</div>

### محاكاة مجموعة الكلمات من متجهات الكلمات (N-Gram using CNN)
{: .text-right}

<div dir='rtl'>
تكلمنا في المقال السابق عن كيف ان استخدام اكثر من كلمة كوحدة للمعالجة (N-grams) يساعد على فهم معلومات اكثر في النص لان بعض المصطلحات يشمل اكثر من كلمة مثل مصطلح (حسبي الله  و نعم الوكيل) على سبيل المثال.
</div>

<div dir='rtl'>
يمكننا استخدام ال CNN لتحصيل المعلومات التي تشتمل عليها مجموعة كلمات من متجهاتها، باستخدام طبيعة ال CNN الالتفافية يمكننا ان نمر على اكثر من كلمة باستخدام نافذة متحركة بحجم معين (مثلا 3 للتعبير عن 3-grams) كما هو موضح في المثال بالاسفل.
</div>

![cnn-n-grams](https://cezannec.github.io/assets/cnn_text/conv_maxpooling_steps.gif){: .align-center}
<div dir='rtl'>
<center><a href='https://cezannec.github.io/CNN_Text_Classification/'>المصدر</a></center></div>


### تعلم متجهات الاحرف (char embeddings)
{: .text-right}

<div dir='rtl'>
يمكن ايضا استخدام ال CNN على مستوى الحروف لتعلم ما تعبر عنه الحروف و بالتالي ببناء اكثر من طبقة متتالية من نوعية CNN يمكن تكوين معرفة عن النص ككل، فكل طبقة/مرحلة سوف تتعلم مما تعلمته السابقة.
</div>

<div dir='rtl' class='notice--info'>
استخدام الحروف بدلا للكلمات يساعد على تفادي عدم معرفة الكلمات الجديدة، ويساعد ايضا على الاستفادة من التكوين المشترك لبعض الكلمات و الاجزاء المشتركة على سبيل المثال يمكننا ان ندرك ان يحب و احب و يحبوا يشتركوا في حرفي الحاء و الباء و لذلك فان الكلمات المشتقة منها يمكن للنوذج معرفتها على عكس النماذج التي تعتمد على الكلمات كوحدة للمعالجة اذ ان الكلمة اذا كانت جديدة شكليا بزيادة حرف على سبيل المثال لن يعرفها النموذج، هذه الطريقة تم استخدامها في نموذج <a href='https://fasttext.cc/'>fasttext</a>  الخاص بشركة facebook، حيث انه استخدم بالاساس مجموعة حروف بدل من كلمات محددة و قد اظهر تحسن ملحوظ عن استخدام الكلمات فقط في معمارية CBOW و skip-gram لاستخراج المتجهات.
</div>

<div dir='rtl'>
</div>

![char-embeddings-cnn](https://www.researchgate.net/profile/Hoang-Pham-17/publication/316875275/figure/fig1/AS:493031154556929@1494559195862/The-CNN-for-extracting-character-level-features-of-word-Hoc-sinh-Student.png){: .align-center}
<div dir='rtl'>
<center><a href='https://www.researchgate.net/figure/The-CNN-for-extracting-character-level-features-of-word-Hoc-sinh-Student_fig1_316875275'>المصدر</a></center></div>


## محاكاة البيانات التسلسلية (Sequential modeling)
{: .text-right}

<div dir='rtl'>
تعد البيانات النصية بيانات تسلسلية، إذ ان ترتيب الكلمات قد يغير المعنى تماما و بالتالي لكي يستطيع نموذج تعلم ما يحتويه النص لابد ان يأخذ في الاعتبار التسلسل و ترتيب الكلمات، و هذا ما لم يفعله ايا من النماذج السابق ذكرها، كما ترى في المثال هنا ان استخدام الشبكة العصبية التقليدية (ANN) ليس لديه اي معلومة عن ترتيب النص لان تغير النص لن يغير في معادلته شئ.
</div>

![ann-words]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/ann.png){: .align-center}
<center>استخدام شبكة عصبية مع الكلمات بفرض استخدام الكلمات كوحدة معالجة</center>

$$h1('product','was','not','good')=$$

<br/>
$$W_{1,1}*'product'+W_{1,2}*'was'+W_{1,3}*'not'+ W_{1,4}*'good'$$

<div dir='rtl'>
لاحظ هنا ان اختلاف الترتيب لن يغير في المعني شئ لانه لا توجد آلية لتعلم العلاقة التي تنشئ من ترتيب الكلمات بشكل معين.
</div>

<div dir='rtl'>
يمكننا تعديل هذه المعمارية بإضافة آلية لتعلم العلاقة التي تنشأ من ترتيب النص كما يلي
</div>

![rnn-words]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/rnn.png){: .align-center}
<center>الشبكة العصبية التكرارية (Recurrent Neural Network)</center>

$$h1(x) = {\color{Red} h1('good', } {\color{Green} h1('not',} {\color{Blue} h1('was', } {\color{Yellow} h1('product', '<sos>')}{\color{Blue} )}{\color{Green} )}{\color{Red} )}$$

<div dir='rtl'>
كما نرى هنا كل خطوة في هذه المعمارية تعتمد على الخطوة السابقة لها بالتالي يمكن للنموذج ان يتعلم ما يعنيه ترتيب الكلمات بشكل افضل.
</div>

<div dir='rtl' class='notice--info'>
الشبكة التكرارية تعمل عن طريق معالجة النصوص بالتتالي، اي ان نفس وحدة المعالجة (الخلية/cell) تقوم بمعالجة الكلمة الاولى ثم الكلمة الثانية و هكذا و لاحظ هنا ان نفس الوحدة تعالج الكلمات بالتتالي و هذا يمنحها القدرة على تعلم العلاقة الموجودة بين الكلمات بناءا على ترتيبهم.
<br>

<center>
<img src='https://www.outsystems.com/blog/-/media/images/blog/posts/graph-neural-networks/nn-gif-2.gif?h=393&w=750&updated=20220111091724'/></center>
<center><a href='https://www.outsystems.com/blog/posts/graph-neural-networks/'>المصدر</a></center>

</div>

<div dir='rtl'>
تقوم هذه المعمارية ببناء فكرة عن النص في كل خطوة زمنية، و هنا الخطوة الزمنية تعني ترتيب الكلمات اي ان الخطوة الزمنية الاولى هي الكلمة الاولى و هكذا، ويمكننا القول ان عند اي نقطة زمنية تقوم الشبكة بتكوين معلومات عن النص حتى هذه النقطة في الزمن و بالتالي عند اخر كلمة ستكون الشبكة كونت فكرة عن النص ككل.
</div>

<br>

<div dir='rtl'>
يمكن استخدام هذه المعمارية في حل العديد من مسائل معالجة اللغة كما يلي.
</div>

![rnn-different-approaches](https://camo.githubusercontent.com/dd88a4672c10fc5e3d6fcf33bcbc4f142f14bbdd3929aa7a336b8c50bc7033fb/687474703a2f2f6b617270617468792e6769746875622e696f2f6173736574732f726e6e2f64696167732e6a706567)
<center><a href='http://karpathy.github.io/2015/05/21/rnn-effectiveness/'>المصدر</a></center>

<div dir='rtl'>
الشكل السابق يمثل 5 انواع من مسائل معالجة اللغة
<ul>
<li>الاول (one-to-one) يمثل الطريقة غير التسلسلية حيث حجم المدخلات و المخرجات ثابت.</li>
<li>الثاني (one-to-many) عندما نستخدم مدخل واحد مقابل اكثر من نتيجة مثل وصف الصور حيث يكون المدخل هو الصورة و النتيجة هي النص الذي يصف الكلمة و بالطبع يمكن ان يكون اكثر من كلمة.</li>
<li>الثالث (many-to-one) يعبر عن اكثر من مدخل (قطعة نصية على سبيل المثال) مقابل نتيجة واحدة (تصنيف النص اذا كان ايجابي او سلبي على سبيل المثال).</li>
<li>الرابع (many-to-many) هنا يتم استهلاك المدخلات كلها قبل ان يتم بناء النتيجة والامثلة على هذا النوع تشمل (الترجمة الآلية و تلخيص النصوص و غيرها).</li>
<li>الخامس (many-to-many) مثل السابق ولكن هنا يوجد نتيجة مباشرة لكل وحدة مدخلة (الوحدة قد تكون كلمة او حرف او مجموعة حروف (character n-grams) على حسب تعريف المدخلات) و احد الامثلة على هذا النوع هو التعرف على الاسماء المعرفة في النص (Named Entity Recognition).</li>
</ul>
</div>

### كيفية عمل الشبكة التكرارية (RNN)
{: .text-right}

<div dir='rtl'>
تعمل الشبكة التكرارية على تعلم العلاقة التي تنشأ من ترتيب النصوص عن طريق تعلم مجموعة من المتغيرات التي تحدد ما الذي يجب تعلمه من النص الحالي و ما الذي يجب تعلمه من الخطوة السابقة (t-1)، كما نرى في الرسم الآتي:
<ol>
<li>ان الكلمة الحالية (Very) يتم معالجتها لتوليد متجه يعبر عنها (Very vector).</li>
<li>يتم معالجة هذا المتجه من خلال المصفوفة X(t) و هي المسئولة عن اضافة ما يجب اضافته و ازالة ما لا تحتاجه الشبكة لتعلم ما يعنيه النص.</li>
<li>الخطوة السابقة للشبكة (RNN Hidden state t-1) التي تمثل المعالجة الناتجة عن الكلمات السابقة يتم معالجتها من خلال مصفوفة اخرى a(t-1) حيث يكون هدفها تعلم ما يجب ان يتم اضافته من الخطوات السابقة و ما يجب ان يتم ازالته.</li>
<li>يتم الجمع بين الخطوتين 2 و 3 و معالجتهم من خلال مرحلة غير خطية تمثلها هنا الدالة tanh و ينتج عنها متجه يعبر عن حالة/فهم الشبكة للنص حتى هذه اللحظة (RNN Hidden state t).</li>
</ol>
</div>

![inside-rnn-1]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/inside-rnn-1.png){: .align-center}
<center><div dir='rtl'>معالجة الكلمة الاولى من النص (Very)</div></center>
<br>

<div dir='rtl'>
ثم يتم تكرير الخطوات السابقة باستخدام نفس المصفوفات X(t) و a(t-1) مع الكلمة التالية كما هو موضح بالشكل الاتي
</div>

![inside-rnn-2]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/inside-rnn-2.png){: .align-center}
<center><div dir='rtl'>معالجة الكلمة الثانية من النص (good)</div></center>

<br>

<div dir='rtl'>
هذه العملية تسمح للشبكة بعد التدريب و تعديل قيم متغيراتها باستخدام النزول التدريجي و تصحيح اخطائها بان تتعلم القيم المناسبة للمصفوفات الخاصة بها لكي تستطيع تعلم النص بشكل يناسب المسئلة التي نقوم بحلها.
</div>

### الشبكة التكرارية ذات الابواب (Gated Recurrent Unit)
{: .text-right}

<div dir='rtl'>
الوصف السابق للشبكة التكرارية يعاني من مشكلة اساسية و هي ان مع طول النص سيكون من الصعب على الشبكة تذكر ما كان في بداية الجملة لان كما نرى في آلية العمل هنا ان الكلمة الحالية و الحالة السابقة لديهم قدرة كبيرة على تغيير حالة الشبكة ولا يوجد آلية للابقاء على المعلومات السابقة في النص.
</div>

<div dir='rtl'>
لذا تم استحداث نموذج معدل من الشبكة التكرارية هو الشبكة التكرارية ذات الابواب (Gated Recurrent Unit GRU) و هي كما يقول الاسم عبارة عن شبكة تكرارية ولكن مع اضافة بوابات، تقوم هذه البوابات بالسماح للشبكة بالحفاظ على معلومات من خطوات سابقة عن طريق تعلم مجموعة متغيرات جديدة تكون البوابة التي تتحكم في مرور او عدم مرور المعلومات الناتجة من الكلمة الحالية و الكلمات السابقة كما هو موضح بالشكل الاتي.
</div>

![gru-1]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/gru-1.png){: .align-center}
<center><div dir='rtl'>GRU</div></center>


<div dir='rtl'>
لفهم اعمق لما يحدث هنا دعنا نلقي نظرة على المعادلات لهذا المجسم
</div>

$$\tilde{C} = tanh(W_c [C^{t-1}, X^t] + b_c)$$

<div dir='rtl'>
هذه المعادلة تمثل الشبكة التكرارية العادية (Vanilla RNN)
</div>

$$\Gamma_u = \sigma(W_u [C^{t-1}, X^t] + b_u)$$

<div dir='rtl'>
هنا يتم تعلم البوابة التي تتحكم في مرور المعلومات من الكلمة الجديدة و الكلمات السابقة، و كما ترى هنا فهي معادلة تعتمد على الكلمة الحالية Xt و الحالة السابقة للشبكة C(t-1)
</div>

![gru-2]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/gru-2.png){: .align-center}
<center><div dir='rtl'>عند هذه النقطة لدينا المتجه Ct الذي يعبر عن الحالة الجديدة من الكلمة الحالية و كذلك المتجه r الذي سوق يستخدم كبوابة للتحكم في مرور المعلومات</div></center>

$$C^t = \Gamma_u*\tilde{C} + (1-\Gamma_u)* C^{t-1}$$

<div dir='rtl'>
لتكوين الحالة الجدية للشبكة Ctتقوم البوابة في التحكم في مرور المعلومات من الخطوة السابقة و الخطوة الحالية حيث كما نرى الجزء الاول r*C يتحكم في مرور المعلومات من الخطوة الحالية و الجزء الثاني (1-r)*C(t-1) يتحكم في مرور المعلومات من الخطوة السابقة.
</div>

<div dir='rtl'>
لاحظ هنا ان البوابة r تنتج من دالة sigmoid اي ان قيمها ستكون من 0 ل 1 بالتالي يمكنك التفكير فيها على انها تمنع قيم و تمرر قيم كما ترى في الشكل التالي.
</div>

![gru-3]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/gru-3.png){: .align-center}
<center><div dir='rtl'>ما يتم ازالته من متجه يتم تعويضه من المتجه الاخر</div></center>

<div dir='rtl'>
تسمح هذه المعمارية بالاحتفاظ بمعلومات سابقة على عكس الشبكة التكرارية التقليدية، لهذا يمكنها معالجة نصوص اطول.
</div>

### الشبكة ذات الذاكرة الطويلة و القصيرة المدى (Long Short Term Memory - LSTM)
{: .text-right}

<div dir='rtl'>
استخدام البوابات في التحكم في مرور المعلومات يسمح للشبكة بتعلم النصوص الطويلة نسبيا و على غرار الشبكة ذات البوابات تم ايضا بناء نوع اخر من الشبكة التكرارية و هي الشبكة ذات الذاكرة الطويلة و القصيرة المدى (Long Short Term Memory - LSTM)، و تقوم هذا المعمارية بإضافة اكثر من بوابة للتحكم في مرور المعلومات و تعمل على اضافة ذاكرة طويلة المدى للحفاظ على المعلومات القديمة و كذلك ذاكرة قصيرة للتعبير عن المعلومات القريبة في النص.
</div>

<div dir='rtl'>
المبدأ المستخدم في الشبكة ذات الذاكرة لايختلف كثيرا عن ذات البوابات، حيث يوجد في الشبكة ذات الذاكرة ثلاث بوابات.
</div>

<div dir='rtl'>
<ul>
<li><b>بوابة النسيان (Forget Gate)</b> تتحكم فيما يتم ازالته او اضافته على الذاكرة الطويلة السابقة (اي الذاكرة الطويلة الناتجة من الخطوة الزمنية السابقة)</li>
<li><b>بوابة التحديث (Update Gate)</b> تتحكم فيما يتم ازالته او اضافته على الذاكرة الطويلة الجديدة (اي الذاكرة الطويلة التي سوف تنتجها الخطوة الزمنية الحالية)</li>
<li><b>بوابة النتيجة (Output Gate)</b> تتحكم في الذاكرة قصيرة المدى الناتجة من الخلية الحالية</li>
</ul>
</div>

![lstm-1]({{ site.url }}{{ site.baseurl }}/assets/images/intro-to-nlp-p2/lstm-1.png){: .align-center}
<center><div dir='rtl'>Long Short Term Memory</div></center>

<br>

<div dir='rtl'>
تعمل البوابات بشكل مشابه لما سبق شرحه، ها هي المعادلات لفهم افضل لما يحدث.
</div>

<br>

بوابة النسيان
{: .text-right}

$$\Gamma_f = \sigma(W_f [a^{t-1}, x^t] + b_f)$$

بوابة التحديث
{: .text-right}

$$\Gamma_u = \sigma(W_u [a^{t-1}, x^t] + b_u)$$

بوابة النتيجة
{: .text-right}

$$\Gamma_o = \sigma(W_o [a^{t-1}, x^t] + b_o)$$

الذاكرة المقترحة C~
{: .text-right}

$$\tilde{C} = tanh(W_c [a^{t-1}, x^t] + b_c) $$

الذاكرة طويلة المدى
{: .text-right}

$$C^t = \Gamma_u*\tilde{C} + \Gamma_f*C^{t-1}$$

<div dir='rtl' class='notice--info'>
لاحظ هنا ان الذاكرة الطويلة الجديدة يتم تكوينها عن طريق الذاكرة الطويلة للخطوة السابقة بعد تمريرها على بوابة النسيان  اضافة الي الذاكرة الجديدة المقترحة بعد تمريرها على بوابة التحديث، وهذا يسمح للشبكة بالتحكم فيما يتم تمريره من الذاكرة السابقة و ما يتم تمريره من الذاكرة المقترحة.
</div>

الذاكرة قصيرة المدى
{: .text-right}

$$a^t = \Gamma_o*tanh(C^t)$$

### تعديلات الشبكة التكرارية
{: .text-right}

<div dir='rtl'>
كأي شبكة عصبية يمكننا اضافة اكثر من طبقة من نفس النموذج لزيادة تعقيد النموذج و بالتالي تمكينه من تعلم انماط اكثر تعقيدا و لكن الشبكة التكرارية بطبيعة الحال تعاني من كونها تكرارية و يجب على كل الكلمات ان تمر بالشبكة كلمة تلو الاخرى مما يعني ان لمعالجة الكلمة الثانية في الطبقة الثانية من الشبكة يجب على كل الكلمات السابقة في نفس الطبقة ان تكون انتهت من المعالجة و كذلك كل الكلمات السابقة في الطبقات السابقة لها مما يجعل من الصعب تحقيق اقصى استفادة من الموارد المتاحة لتدريب نماذج اكبر. 
</div>

<br>

![2-layer-rnn](https://static.packt-cdn.com/products/9781787121089/graphics/image_06_008.png){: .align-center}
<center><a href='https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781787121089/6/ch06lvl1sec70/setting-up-a-deep-rnn-model'>المصدر</a></center>

<br>

<div dir='rtl'>
يمكننا ايضا جعل الشبكة التكرارية ترى في الاتجاهيين، بمعنى ان في اي لحظة زمنية يمكن للشبكة معرفة الكلمات السابقة و الكلمات القادمة و هذا يمكن ان يكون مفيد للغاية في تطبيقات مثل التعرف على الاسماء المعرفة (NER) على سبيل المثال، و يسمى هذا النوع من الشبكة بالشبكة التكرارية ثنائية الاتجاه (Bi-Directional RNN).
</div>

<div dir='rtl'>
لتحقيق هذا يمكننا استخدام شبكتين تكراريتين، واحدة تعالج النص من اليمين لليسار و الاخرى تقوم بالعكس، وعند معالجة الكلمة نستخدم مجموع ما تعلمته الشبكة و الاولى و الشبكة الثانية مما يضيف ثراء معلوماتي للكلمة.
</div>

<br>

![bi-directional-rnn](https://miro.medium.com/max/764/1*6QnPUSv_t9BY9Fv8_aLb-Q.png){: .align-center}
<center><a href='http://colah.github.io/posts/2015-09-NN-Types-FP/'>المصدر</a></center>


<div dir='rtl' class='notice--info'>
لاحظ هنا ان كل شبكة لاترى النص ككل و انما تراه جزء فقط، اي ان الشبكة التي تعمل من اليسار لاترى الا الكلمات السابقة لها ولا ترى الكلمات التي سوف تأتي لاحقأ وكذلك الشبكة التي تعمل من اليمين لاترى الا الكلمات يمينها ولا ترى الكلمات في اليسار، لذلك يعتبر استخدام النموذج ثنائي الاتجاه تحايل على رؤية النص كله ولكن فعليا هذا لايحدث هنا بشكل كامل.
</div>

### حدود الشبكة التكرارية
{: .text-right}

<div dir='rtl'>
بسبب طبيعتها التكرارية كما ذكرنا من قبل فان الشبكة التكرارية تعاني عدم القدرة على الاستغلال الأمثل للموارد بمعنى انه لا يمكننا تدريب طبقات هذه الشبكة بالتوازي لان الخطوات الزمنية تعتمد على بعضها البعض لذا يجب على كل خطوة انتظار الخطوة السابقة لها مما يمنع تدريبها بشكل متوازي.
</div>

<div dir='rtl'>
ايضا الشبكة لا تر النص بشكل كامل كما ذكرنا حتى مع استخدام ثنائية الاتجاه لان كما ذكرنا كل شبكة ترى فقط الكلمات السابقة على حسب اتجاهها.
</div>

<div dir='rtl'>
وعلى الرغم من استخدام البوابات و الذاكرة للاحتفاظ بالمعلومات الا انه من الصعب الاحتفاظ بكل المعلومات مع طول النص المعالج لذلك يتم فقد جزء من معلومات النص.
</div>

## الاستنتاج
{: .text-right}

<div dir='rtl'>
في هذا المقال تحدثنا عن مبدأ ال word embeddings او تمثيل النصوص باستخدام متجهات تعبر عن المعنى و التي كانت قفزة كبيرة في عالم معالجة اللغة و ايضا تكلمنا عن طريقة عملها و طرق مختلفة في تطبيقها.
</div>

<div dir='rtl'>
تناولنا ايضا استخدام معماريات مختلفة مثل CNN و LSTM و كيفية عملهم و الصعوبات التي تواجه بعض النماذج.
</div>

<div dir='rtl'>
في المقال القادم سوف نتعرض لمعماريات (architectures) جديدة استطاعت ان تتغلب على ما سبقها في عدة مهام مما جعلها احدث ما ورد في المجال (State of the Art) حتى تاريخ كتابة هذا المقال 
</div>


<div dir='rtl' class='notice--success'>
في هذا المقال حاولت تبسيط بعض المصطلحات للغتنا العربية من اجل تسهيل عملية الشرح ولتبسيط المعلومة، في حالة اي خطأ املائي او اقتراح افضل للترجمة فأنا ارحب جدا بذلك يمكنك التعليق على المقال او مراسلتي لتعديل و تحسين المحتوى، ووفقنا الله وإياكم لما يحب ويرضى.
</div>

## مصادر
{: .text-right}

<div dir='rtl'>
<ul>
<li><a href='https://arxiv.org/abs/1301.3781'>الورقة البحثية لنموذج word2vec</a></li>
<li><a href='https://www.coursera.org/learn/nlp-sequence-models'>كورس Sequence Models من deeplearning.ai</a></li>
</ul>
</div>
