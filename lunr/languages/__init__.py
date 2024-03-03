from itertools import chain
from functools import partial

import lunr
from lunr.builder import Builder
from lunr.languages.trimmer import generate_trimmer
from lunr.languages.stemmer import nltk_stemmer, get_language_stemmer
from lunr.pipeline import Pipeline
from lunr.stop_word_filter import stop_word_filter, generate_stop_word_filter

# map from ISO-639-1 codes to SnowballStemmer.languages
# Languages not supported by nltk but by lunr.js: thai, japanese and turkish
# Languages upported by nltk but not lunr.js: arabic

SUPPORTED_LANGUAGES = {
    "fa": "arabic",
    "ar": "arabic",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "hu": "hungarian",
    "it": "italian",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "es": "spanish",
    "sv": "swedish",
}
persian_stopwords={c for c in 'و در به از كه مي اين است را با هاي براي آن يك شود شده خود ها كرد شد اي تا كند بر بود گفت نيز وي هم كنند دارد ما كرده يا اما بايد دو اند هر خواهد او مورد آنها باشد ديگر مردم نمي بين پيش پس اگر همه صورت يكي هستند بي من دهد هزار نيست استفاده داد داشته راه داشت چه همچنين كردند داده بوده دارند همين ميليون سوي شوند بيشتر بسيار روي گرفته هايي تواند اول نام هيچ چند جديد بيش شدن كردن كنيم نشان حتي اينكه ولی توسط چنين برخي نه ديروز دوم درباره بعد مختلف گيرد شما گفته آنان بار طور گرفت دهند گذاري بسياري طي بودند ميليارد بدون تمام كل تر  براساس شدند ترين امروز باشند ندارد چون قابل گويد ديگري همان خواهند قبل آمده اكنون تحت طريق گيري جاي هنوز چرا البته كنيد سازي سوم كنم بلكه زير توانند ضمن فقط بودن حق آيد وقتي اش يابد نخستين مقابل خدمات امسال تاكنون مانند تازه آورد فكر آنچه نخست نشده شايد چهار جريان پنج ساخته زيرا نزديك برداري كسي ريزي رفت گردد مثل آمد ام بهترين دانست كمتر دادن تمامي جلوگيري بيشتري ايم ناشي چيزي آنكه بالا بنابراين ايشان بعضي دادند داشتند برخوردار نخواهد هنگام نبايد غير نبود ديده وگو داريم چگونه بندي خواست فوق ده نوعي هستيم ديگران همچنان سراسر ندارند گروهي سعي روزهاي آنجا يكديگر كردم بيست بروز سپس رفته آورده نمايد باشيم گويند زياد خويش همواره گذاشته شش  نداشته شناسي خواهيم آباد داشتن نظير همچون باره نكرده شان سابق هفت دانند جايي بی جز زیرِ رویِ سریِ تویِ جلویِ پیشِ عقبِ بالایِ خارجِ وسطِ بیرونِ سویِ کنارِ پاعینِ نزدِ نزدیکِ دنبالِ حدودِ برابرِ طبقِ مانندِ ضدِّ هنگامِ برایِ مثلِ بارة اثرِ تولِ علّتِ سمتِ عنوانِ قصدِ روب جدا کی که چیست هست کجا کجاست کَی چطور کدام آیا مگر چندین یک چیزی دیگر کسی بعری هیچ چیز جا کس هرگز یا تنها بلکه خیاه بله بلی آره آری مرسی البتّه لطفاً ّه انکه وقتیکه همین پیش مدّتی هنگامی مان تان'.split(" ")};
try:  # pragma: no cover
    import nltk  # type: ignore

    LANGUAGE_SUPPORT = True
except ImportError:  # pragma: no cover
    LANGUAGE_SUPPORT = False


def _get_stopwords_and_word_characters(language):
    nltk.download("stopwords", quiet=True)
    verbose_language = SUPPORTED_LANGUAGES[language]
    stopwords = nltk.corpus.stopwords.words(verbose_language)
    # TODO: search for a more exhaustive list of word characters
    if language == 'fa':
        #stopwords={*stopwords, *persian_stopwords}
        stopwords=persian_stopwords
    word_characters = {c for word in stopwords for c in word}
    
    return stopwords, word_characters


def get_nltk_builder(languages):
    """Returns a builder with stemmers for all languages added to it.

    Args:
        languages (list): A list of supported languages.
    """
    all_stemmers = []
    all_stopwords_filters = []
    all_word_characters = set()

    for language in languages:
        if language == "en":
            # use Lunr's defaults
            all_stemmers.append(lunr.stemmer.stemmer)
            all_stopwords_filters.append(stop_word_filter)
            all_word_characters.update({r"\w"})
        else:
            stopwords, word_characters = _get_stopwords_and_word_characters(language)
            all_stemmers.append(
                Pipeline.registered_functions["stemmer-{}".format(language)]
            )
            all_stopwords_filters.append(
                generate_stop_word_filter(stopwords, language=language)
            )
            all_word_characters.update(word_characters)

    builder = Builder()
    multi_trimmer = generate_trimmer("".join(sorted(all_word_characters)))
    Pipeline.register_function(
        multi_trimmer, "lunr-multi-trimmer-{}".format("-".join(languages))
    )
    builder.pipeline.reset()

    for fn in chain([multi_trimmer], all_stopwords_filters, all_stemmers):
        builder.pipeline.add(fn)
    for fn in all_stemmers:
        builder.search_pipeline.add(fn)

    return builder


def register_languages():
    """Register all supported languages to ensure compatibility."""
    for language in set(SUPPORTED_LANGUAGES) - {"en"}:
        language_stemmer = partial(nltk_stemmer, get_language_stemmer(language))
        Pipeline.register_function(language_stemmer, "stemmer-{}".format(language))


if LANGUAGE_SUPPORT:  # pragma: no cover
    # TODO: registering all possible stemmers feels unnecessary but it solves
    # deserializing with arbitrary language functions. Ideally the schema would
    # provide the language(s) for the index and we could register the stemmers
    # as needed
    register_languages()
