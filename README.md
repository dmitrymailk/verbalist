# Verbalist (буквоед) - русскоязычный ассистент.

Проект во многом вдохновленный [Saiga](https://huggingface.co/IlyaGusev/saiga2_7b_lora).

Мною были собраны все самые качественные датасеты с [huggingface.datasets](https://huggingface.co/datasets), а также собраны дополнительно с тех сайтов, которые я посчитал весьма полезными для создания аналога ChatGPT. Лицензии у всех датасетов отличаются, какие-то по типу [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) были созданы специально для обучения подобных моделей, какие-то являются прямой выгрузкой диалогов с ChatGPT ([RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)).

Вклад данного репозитория состоит в систематизации и стандартизации уже имеющихся датасетов, добавлении новых. А также тренировке моделей на этих данных.

- [google sheets таблица с датасетами и описанием](https://docs.google.com/spreadsheets/d/10xcsINF_c_zUZchT8p-8xIuHDgcuwg63jjl2ortBP9I/edit?usp=sharing)

### Датасеты

- **[Объединенный датасет где все данные уже подготовлены для тренировки диалоговой модели](https://huggingface.co/datasets/dim/verbalist_prompts)**
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">name</th>
<th title="Field #2">link</th>
<th title="Field #3">description</th>
<th title="Field #4">original_name</th>
<th title="Field #5">original_source</th>
<th title="Field #6">preparation_script</th>
<th title="Field #7">language</th>
</tr></thead>
<tbody><tr>
<td>dim/oasst_en</td>
<td>https://huggingface.co/datasets/dim/oasst_en</td>
<td>OpenAssistant Conversations Dataset на английском языке, профильтрованный вручную мной. В оригинальном датасете около 30% диалогов не являются корректными. Иногда пользователь который отыгрывает ассистента грубит пользователю, иногда люди просто отвечали не знаю на вопросы, некоторые вопросы были недостаточно научными или слишком короткими. Данную разметку вы можете посмотреть посмотреть по этой ссылке https://docs.google.com/spreadsheets/d/117t5-Tr-dxdODpyFBkBg5R8GklYBlsvBfeDyjqwz2pA/edit?usp=sharing</td>
<td>2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/oasst</td>
<td>en</td>
</tr>
<tr>
<td>dim/oasst_ru</td>
<td>https://huggingface.co/datasets/dim/oasst_ru</td>
<td>OpenAssistant Conversations Dataset на русском языке, профильтрованный вручную мной. В оригинальном датасете около 30% диалогов не являются корректными. Иногда пользователь который отыгрывает ассистента грубит пользователю, иногда люди просто отвечали не знаю на вопросы, некоторые вопросы были недостаточно научными или слишком короткими. Данную разметку вы можете посмотреть посмотреть по этой ссылке https://docs.google.com/spreadsheets/d/1uiOnqxiytuxrB6u6q2pMSdnMfqjT3arfg8DlT-OWlb0/edit?usp=sharing</td>
<td>2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/oasst</td>
<td>ru</td>
</tr>
<tr>
<td>dim/lima</td>
<td>https://huggingface.co/datasets/dim/lima</td>
<td>Датасет состоящий и 1000 качественных обучающих примеров на английском языке. Stack Exchange (STEM), Stack Exchange (Other), wikiHow, Pushshift r/WritingPrompts, Natural Instructions, уникальные инструкции составленные авторами статьи. Больше об этом можно прочитать тут https://arxiv.org/pdf/2305.11206.pdf</td>
<td>GAIR/lima</td>
<td>https://huggingface.co/datasets/GAIR/lima</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/lima</td>
<td>en</td>
</tr>
<tr>
<td>dim/logic_tasks_ru</td>
<td>https://huggingface.co/datasets/dim/logic_tasks_ru</td>
<td>Детские задачки на логику с сайта https://www.potehechas.ru/zadachi/zadachi.shtml</td>
<td>Логические задачи - Логика и нестандартное мышление</td>
<td>https://www.potehechas.ru/zadachi/zadachi.shtml</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/logic_tasks_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/wikihow_en</td>
<td>https://huggingface.co/datasets/dim/wikihow_en</td>
<td>Англоязычные статьи с сайта wikihow</td>
<td>0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/wiki_how</td>
<td>en</td>
</tr>
<tr>
<td>dim/wikihow_ru</td>
<td>https://huggingface.co/datasets/dim/wikihow_ru</td>
<td>Русскоязычные статьи с сайта wikihow</td>
<td>0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/wiki_how</td>
<td>ru</td>
</tr>
<tr>
<td>dim/essayforum_writing_prompts_6k</td>
<td>https://huggingface.co/datasets/dim/essayforum_writing_prompts_6k</td>
<td>На данном сайте люди просят помощи с небольшими эссе. На данном сайте может ответить кто угодно, но данный датасет содержит ответы только от главного администратора сайта, так как его ответы чаще всего наиболее качественные и вдумчивые. </td>
<td>EssayForum</td>
<td>https://essayforum.com/writing/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/essayforum</td>
<td>en</td>
</tr>
<tr>
<td>dim/sharegpt_short_ru</td>
<td>https://huggingface.co/datasets/dim/sharegpt_short_ru</td>
<td>Очищенная версия русская версия sharegpt. Я попытался вырезать из текста все промпты, где модель извиняется что что-то не может сделать, что она не имеет доступа в интернет. Диалоги, которые противоречат морали модели я просто исключил. Постарался убрать упоминания о том что она модель AI, так как за ролеплейные характеристики отвечают другие датасеты.</td>
<td>RyokoAI/ShareGPT52K</td>
<td>https://huggingface.co/datasets/RyokoAI/ShareGPT52K</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/sharegpt</td>
<td>ru</td>
</tr>
<tr>
<td>dim/openreview_prompts_65</td>
<td>https://huggingface.co/datasets/dim/openreview_prompts_65</td>
<td>Датасет рецензий на реальные научные статьи с сайта openreview. Вышло на самом деле не так много, так как многие статьи не выложенны на arxiv или просто не имеют рецензий. Плюс я собрал только малую часть данного сайта, а не все что там было. </td>
<td>https://openreview.net/</td>
<td>https://openreview.net/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/openreview</td>
<td>en</td>
</tr>
<tr>
<td>dim/roleplay_instruct_v2_final</td>
<td>https://huggingface.co/datasets/dim/roleplay_instruct_v2_final</td>
<td>Датасет ролеплея от GPT-4 на различных персонажей на английском языке.</td>
<td>roleplay-instruct-v2-final</td>
<td>https://github.com/teknium1/GPTeacher</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/gpt_roleplay_realm</td>
<td>en</td>
</tr>
<tr>
<td>dim/kinomania_scripts</td>
<td>https://huggingface.co/datasets/dim/kinomania_scripts</td>
<td>Небольшой датасет, который содержит в себе сценарии фильмов целиком и их краткое содержание</td>
<td>https://www.kinomania.ru/scripts</td>
<td>https://www.kinomania.ru/scripts</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/kinomania_scripts</td>
<td>ru\en</td>
</tr>
<tr>
<td>dim/bugurt_thread_prompts</td>
<td>https://huggingface.co/datasets/dim/bugurt_thread_prompts</td>
<td>Небольшой набор размеченных бугуртов вместе с моим другом, для того чтобы модель научилась писать бугурты на конкретную ситуацию. Собраны из телеграм паблика БУГУРТ ТРЕД(https://t.me/bugurtthread)</td>
<td>https://t.me/bugurtthread</td>
<td>https://t.me/bugurtthread</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/bugurt_thread</td>
<td>ru</td>
</tr>
<tr>
<td>dim/russian_lyrics_prompts</td>
<td>https://huggingface.co/datasets/dim/russian_lyrics_prompts</td>
<td>Небольшой датасет промптов собранный мною из различных учебников по стихосложению, чтобы модель научилась писать стихи, используя необходимый литературный прием на конкретную тему.</td>
<td>Учебник стихосложения</td>
<td>https://stihi.ru/uchebnik/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/russian_lyrics_prompts</td>
<td>ru</td>
</tr>
<tr>
<td>dim/ru_instruct_gpt4</td>
<td>https://huggingface.co/datasets/dim/ru_instruct_gpt4</td>
<td>Датасет каких-то инструкций на русском сгенерированных GPT-4</td>
<td>lksy/ru_instruct_gpt4</td>
<td>https://huggingface.co/datasets/lksy/ru_instruct_gpt4</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ru_instruct_gpt4</td>
<td>ru</td>
</tr>
<tr>
<td>dim/gpt_roleplay_realm</td>
<td>https://huggingface.co/datasets/dim/gpt_roleplay_realm</td>
<td>Диалоги выдуманных персонажей при помощи GPT-4, диалоги были сгенерированны при помощи GPT-3.5. Русский и английский.</td>
<td>IlyaGusev/gpt_roleplay_realm</td>
<td>https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/gpt_roleplay_realm</td>
<td>ru\en</td>
</tr>
<tr>
<td>dim/ultrachat_ru</td>
<td>https://huggingface.co/datasets/dim/ultrachat_ru</td>
<td>Какой-то рандомный датасет диалогов от chatgpt, который я нашел на huggingface. Из текста диалогов были вырезаны шаблонные фразы по типу: &quot;я не могу выполнить&quot;, &quot;как языковая модель&quot; и тд. Потому что обычно после этого следовало вменяемое решение задачи.</td>
<td>kaleinaNyan/UltraChat_ru</td>
<td>https://huggingface.co/datasets/kaleinaNyan/UltraChat_ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ultrachat_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/scitldr</td>
<td>https://huggingface.co/datasets/dim/scitldr</td>
<td>Саммаризация научных статей на английском языке, выполненная экспертами.</td>
<td>allenai/scitldr</td>
<td>https://huggingface.co/datasets/allenai/scitldr</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/scitldr</td>
<td>en</td>
</tr>
<tr>
<td>dim/linux_man_pages_tldr_summarized</td>
<td>https://huggingface.co/datasets/dim/linux_man_pages_tldr_summarized</td>
<td>Саммаризация мануалов для инструментов линукс в удобный набор команд с их кратким описанием.</td>
<td>tmskss/linux-man-pages-tldr-summarized</td>
<td>https://huggingface.co/datasets/tmskss/linux-man-pages-tldr-summarized</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/linux-man-pages-tldr-summarized</td>
<td>en</td>
</tr>
<tr>
<td>dim/dolphin_ru_3k</td>
<td>https://huggingface.co/datasets/dim/dolphin_ru_3k</td>
<td>Подвыборка размера 3000 переведенных заданий dolphin. Примеры из оригинального датасета это промпты из FLANv2 и решения при помощи GPT-4 или GPT-3.5.</td>
<td>d0rj/dolphin-ru</td>
<td>https://huggingface.co/datasets/d0rj/dolphin-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/dolphin_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/runne_prompts</td>
<td>https://huggingface.co/datasets/dim/runne_prompts</td>
<td>Промпты составленные из датасета RuNNE. Лично я при обучении сотавил промпт следующим образом. Сначала идет текст &quot;Найди все именованные сущности в данном тексте:&quot;, а затем шел сам текст. В качестве выхода модели нужно сгенерировать JSON где содержатся все найденные именованные сущности. К примеру так [{&quot;name&quot;: &quot;PERSON&quot;, &quot;ent&quot;: &quot;Ким Чен Нама&quot;, &quot;pos&quot;: &quot;0 12&quot;}, {&quot;name&quot;: &quot;ORGANIZATION&quot;, &quot;ent&quot;: &quot;Полиция Малайзии&quot;, &quot;pos&quot;: &quot;56 72&quot;}]</td>
<td>iluvvatar/RuNNE</td>
<td>https://huggingface.co/datasets/iluvvatar/RuNNE</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/RuNNE</td>
<td>ru</td>
</tr>
<tr>
<td>dim/lurk_prompts</td>
<td>https://huggingface.co/datasets/dim/lurk_prompts</td>
<td>Набор определений различных терминов с сайта lurk. Сами промпты были составлены автоматически следующим образом. напиши определение для (ОПРЕДЕЛЕНИЕ) в стиле lurk</td>
<td>averoo/lurk</td>
<td>https://huggingface.co/datasets/averoo/lurk/viewer/default/train?p=2</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/lurk</td>
<td>ru</td>
</tr>
<tr>
<td>dim/panorama_prompts_10k</td>
<td>https://huggingface.co/datasets/dim/panorama_prompts_10k</td>
<td>Набор юмористических заголовков и текстов новостей с сайта панорама.</td>
<td>its5Q/panorama</td>
<td>https://huggingface.co/datasets/its5Q/panorama</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/panorama</td>
<td>ru</td>
</tr>
<tr>
<td>dim/resh_edu_short_prompts</td>
<td>https://huggingface.co/datasets/dim/resh_edu_short_prompts</td>
<td>Набор уроков с сайта resh.edu.ru включающих в себя название урока, тему, класс и текст урока с заданиями. </td>
<td>its5Q/resh-edu</td>
<td>https://huggingface.co/datasets/its5Q/resh-edu</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/resh_edu</td>
<td>ru</td>
</tr>
<tr>
<td>dim/databricks_dolly_15k_ru</td>
<td>https://huggingface.co/datasets/dim/databricks_dolly_15k_ru</td>
<td>Переведенный датасет dolly на русский язык. Включает в себя набор инструкций на обширное количество тематик.</td>
<td>dwarf2/databricks-dolly-15k-ru</td>
<td>https://huggingface.co/dwarf2/databricks-dolly-15k-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/databricks_dolly_15k_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/databricks_dolly_15k_en</td>
<td>https://huggingface.co/datasets/dim/databricks_dolly_15k_en</td>
<td>databricks-dolly-15k — это набор данных с открытым исходным кодом, содержащий записи о выполнении инструкций, созданные тысячами сотрудников Databricks в нескольких поведенческих категориях, изложенных в документе InstructGPT, включая мозговой штурм, классификацию, закрытый контроль качества, генерацию, извлечение информации, открытый контроль качества и обобщение.</td>
<td>databricks/databricks-dolly-15k</td>
<td>https://huggingface.co/datasets/databricks/databricks-dolly-15k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/databricks_dolly_15k_en</td>
<td>en</td>
</tr>
<tr>
<td>dim/grammarly_coedit</td>
<td>https://huggingface.co/datasets/dim/grammarly_coedit</td>
<td>Набор промптов, которые просят исправить грамматические, стилистические ошибки на английском.</td>
<td>grammarly/coedit</td>
<td>https://huggingface.co/datasets/grammarly/coedit</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/grammarly_coedit</td>
<td>en</td>
</tr>
<tr>
<td>dim/kinopoisk_prompts</td>
<td>https://huggingface.co/datasets/dim/kinopoisk_prompts</td>
<td>Отзывы с кинопоиска на топ 250 фильмов. В промптах я прошу написать хороший, плохой или нейтральный отзыв на определенный фильм.</td>
<td>blinoff/kinopoisk</td>
<td>https://huggingface.co/datasets/blinoff/kinopoisk</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/kinopoisk</td>
<td>ru</td>
</tr>
<tr>
<td>dim/medical_qa_ru_prompts</td>
<td>https://huggingface.co/datasets/dim/medical_qa_ru_prompts</td>
<td>Какие-то вопросы и ответы с какого-то медицинского форума. В данной версии датасета только первый ответ из оригинала.</td>
<td>blinoff/medical_qa_ru_data</td>
<td>https://huggingface.co/datasets/blinoff/medical_qa_ru_data</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/medical_qa_ru_data</td>
<td>ru</td>
</tr>
<tr>
<td>dim/joke_explaination_prompts</td>
<td>https://huggingface.co/datasets/dim/joke_explaination_prompts</td>
<td>Объяснение шуток на английском. От изначального датасета отличается тем, что я убрал последнее предложение из объяснения, так как оно ссылается на видео на сайте.</td>
<td>theblackcat102/joke_explaination</td>
<td>https://huggingface.co/datasets/theblackcat102/joke_explaination</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/joke_explaination</td>
<td>en</td>
</tr>
<tr>
<td>dim/oa_stackexchange_200k</td>
<td>https://huggingface.co/datasets/dim/oa_stackexchange_200k</td>
<td>Вопросы-ответы со stackexchange. Оригинальный датасет был составлен следующим образом: были выбраны только темы с принятым ответом, для которых длина вопроса и ответа составляет менее 1000 символов. Другие ответы, вопросы без принятых ответов или длинные записи были удалены. Так как оригинальный датасет слишком большой, я рандомно выбрал 200k семплов.</td>
<td>donfu/oa-stackexchange</td>
<td>https://huggingface.co/datasets/donfu/oa-stackexchange</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/oa_stackexchange</td>
<td>en</td>
</tr>
<tr>
<td>dim/scale_helpful_no_math</td>
<td>https://huggingface.co/datasets/dim/scale_helpful_no_math</td>
<td>Какой-то набор диалогов с вопросами-ответами на английском, происхождение неизвестно.</td>
<td>HuggingFaceH4/scale_helpful_no_math</td>
<td>https://huggingface.co/datasets/HuggingFaceH4/scale_helpful_no_math/viewer/default/train_rm</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/scale_helpful_no_math</td>
<td>en</td>
</tr>
<tr>
<td>dim/law_stackexchange_prompts</td>
<td>https://huggingface.co/datasets/dim/law_stackexchange_prompts</td>
<td>Вопросы про закон на английском языке со StackExchange. Оригинальный датасет был преобразован в markdown.</td>
<td>ymoslem/Law-StackExchange</td>
<td>https://huggingface.co/datasets/ymoslem/Law-StackExchange</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/law_stackexchange</td>
<td>en</td>
</tr>
<tr>
<td>dim/ficbook_prompts_best_10k</td>
<td>https://huggingface.co/datasets/dim/ficbook_prompts_best_10k</td>
<td>Топ 10k лучших фанфиков с сайта ficbook.net. Все промпты выглядят следующим образом: напиши фанфик с названием {title} и следующим описанием {description}, с тегами {tags}, Где title это оригинальное название, description оригинальное описание, tags это теги данного произведения.</td>
<td>AlexWortega/FicBook</td>
<td>https://huggingface.co/datasets/AlexWortega/FicBook</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ficbook</td>
<td>ru</td>
</tr>
<tr>
<td>dim/azbyka_logic_ru</td>
<td>https://huggingface.co/datasets/dim/azbyka_logic_ru</td>
<td>Небольшой набор детских логических и православных задач, взятых с сайта https://azbyka.ru/deti/logicheskie-i-zanimatelnye-zadachi . Обычно у них почти нет развернутого решения, только ответ. Я пытался расписать решение некоторых задач, но меня хватило только на 35, если кто-то займется подобным буду рад https://docs.google.com/spreadsheets/d/1JRbtppbZCUbV_Eqd0nKbRDQEuPnJIAgJ70cUILEDUI4/edit?usp=sharing . </td>
<td>Логические и занимательные задачи (300 задач)</td>
<td>https://azbyka.ru/deti/logicheskie-i-zanimatelnye-zadachi</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/azbyka_logic_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/povarenok</td>
<td>https://huggingface.co/datasets/dim/povarenok</td>
<td>46k лучших рецептов с сайта povarenok.ru, содержит текст рецепта, список ингридиентов, название блюда</td>
<td>https://www.povarenok.ru/recipes/</td>
<td>https://www.povarenok.ru/recipes/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/povarenok</td>
<td>ru</td>
</tr>
<tr>
<td>dim/AO3_fandom_chatbot_1to1</td>
<td>https://huggingface.co/datasets/dim/AO3_fandom_chatbot_1to1</td>
<td>Какой-то набор ролеплейных диалогов с описанием персонажей и их отыгрышем. Происхождение неизвестно.</td>
<td>ebony59/AO3_fandom_chatbot_1to1</td>
<td>https://huggingface.co/datasets/ebony59/AO3_fandom_chatbot_1to1</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/AO3_fandom_chatbot_1to1</td>
<td>en</td>
</tr>
<tr>
<td>dim/habr_prompts_5k</td>
<td>https://huggingface.co/datasets/dim/habr_prompts_5k</td>
<td>Статьи с хабра. Датасет был составлен с помощью chatgpt, chatgpt преобразовывал заголовки таким образом чтобы они звучали как вопросы от пользователя, в качестве таргета выступала сама статья.</td>
<td>IlyaGusev/habr</td>
<td>https://huggingface.co/datasets/IlyaGusev/habr</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/habr</td>
<td>ru</td>
</tr>
<tr>
<td>dim/what_where_when_50k</td>
<td>https://huggingface.co/datasets/dim/what_where_when_50k</td>
<td>50k вопросов с решениями с сайта что где когда. В качестве промпта выступает вопрос, в качестве ответа конкатенация объяснения и краткого ответа. Все вопросы-ответы вы можете найти по этой ссылке https://huggingface.co/datasets/dim/what_where_when_ru</td>
<td>https://db.chgk.info</td>
<td>https://db.chgk.info</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/what_where_when</td>
<td>ru</td>
</tr>
<tr>
<td>dim/competition_math</td>
<td>https://huggingface.co/datasets/dim/competition_math</td>
<td>Датасет олимпиадной математики на английском. The Mathematics Aptitude Test of Heuristics (MATH) dataset.</td>
<td>competition_math</td>
<td>https://huggingface.co/datasets/competition_math</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/competition_math</td>
<td>en</td>
</tr>
<tr>
<td>dim/sharegpt_short_en_30k</td>
<td>https://huggingface.co/datasets/dim/sharegpt_short_en_30k</td>
<td>Короткие диалоги на английском из sharegpt</td>
<td>RyokoAI/ShareGPT52K</td>
<td>https://huggingface.co/datasets/RyokoAI/ShareGPT52K</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/sharegpt</td>
<td>en</td>
</tr>
<tr>
<td>dim/ru_turbo_alpaca_evol_instruct</td>
<td>https://huggingface.co/datasets/dim/ru_turbo_alpaca_evol_instruct</td>
<td>Набор инструкций различной тематики на русском языке, сгенерированных при помощи chatgpt.</td>
<td>IlyaGusev/ru_turbo_alpaca_evol_instruct</td>
<td>https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca_evol_instruct</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ru_turbo_alpaca_evol_instruct</td>
<td>ru</td>
</tr>
<tr>
<td>dim/ru_turbo_saiga</td>
<td>https://huggingface.co/datasets/dim/ru_turbo_saiga</td>
<td>Набор инструкций различной тематики на русском языке, сгенерированных при помощи chatgpt.</td>
<td>IlyaGusev/ru_turbo_saiga</td>
<td>https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ru_turbo_saiga</td>
<td>ru</td>
</tr>
<tr>
<td>dim/bugurt_completion_prompts</td>
<td>https://huggingface.co/datasets/dim/bugurt_completion_prompts</td>
<td>Обрезанные бугурты, где в качестве промпта используется строка вида - продолжи бугурт: первая строчка бугурта </td>
<td>https://t.me/bugurtthread</td>
<td>https://t.me/bugurtthread</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/bugurt_thread</td>
<td>ru</td>
</tr>
<tr>
<td>dim/tldr_17_50k</td>
<td>https://huggingface.co/datasets/dim/tldr_17_50k</td>
<td>Очень вольная абстрактная саммаризация постов с реддита в одну строчку</td>
<td>webis/tldr-17</td>
<td>https://huggingface.co/datasets/webis/tldr-17</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/tldr_17</td>
<td>en</td>
</tr>
<tr>
<td>dim/grade_school_math_instructions</td>
<td>https://huggingface.co/datasets/dim/grade_school_math_instructions</td>
<td>OpenAI&#39;s grade-school-math датасет преобразованный в промпты.</td>
<td>qwedsacf/grade-school-math-instructions</td>
<td>https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/grade-school-math-instructions</td>
<td>en</td>
</tr>
<tr>
<td>dim/tldr_news</td>
<td>https://huggingface.co/datasets/dim/tldr_news</td>
<td>Хедлайны и текст новостей на различную тематику. </td>
<td>JulesBelveze/tldr_news</td>
<td>https://huggingface.co/datasets/JulesBelveze/tldr_news</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/tldr_news</td>
<td>en</td>
</tr>
<tr>
<td>dim/grade_school_math_instructions_ru</td>
<td>https://huggingface.co/datasets/dim/grade_school_math_instructions_ru</td>
<td>OpenAI&#39;s grade-school-math датасет переведенный на русский.</td>
<td>d0rj/gsm8k-ru</td>
<td>https://huggingface.co/datasets/d0rj/gsm8k-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/grade_school_math_instructions_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/dialogsum</td>
<td>https://huggingface.co/datasets/dim/dialogsum</td>
<td>Саммаризация диалогов на английском языке, разметка выполнялась вручную.</td>
<td>knkarthick/dialogsum</td>
<td>https://huggingface.co/datasets/knkarthick/dialogsum</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/dialogsum</td>
<td>en</td>
</tr>
<tr>
<td>dim/HC3_ru</td>
<td>https://huggingface.co/datasets/dim/HC3_ru</td>
<td>Вопросы-ответы с реддита, есть ответы сгенерированные chatgpt и реальные ответы пользователей. Я использовал только реальные ответы пользователей.</td>
<td>d0rj/HC3-ru</td>
<td>https://huggingface.co/datasets/d0rj/HC3-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/HC3_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/horoscopes_ru_10k</td>
<td>https://huggingface.co/datasets/dim/horoscopes_ru_10k</td>
<td>10k гороскопов, с промптами где я прошу сгенерировать гороском для определенного знака зодиака</td>
<td>dkagramanyan/horoscopes_ru</td>
<td>https://huggingface.co/datasets/dkagramanyan/horoscopes_ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/horoscopes_ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/yandex_q_200k</td>
<td>https://huggingface.co/datasets/dim/yandex_q_200k</td>
<td>200k рандомно выбранных вопросов-ответов с сайта yandex q.</td>
<td>its5Q/yandex-q</td>
<td>https://huggingface.co/datasets/its5Q/yandex-q</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/yandex_q</td>
<td>ru</td>
</tr>
<tr>
<td>dim/leetcodesolutions_en_2k</td>
<td>https://huggingface.co/datasets/dim/leetcodesolutions_en_2k</td>
<td>Решения задач с leetcode на разных языках. </td>
<td>TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k</td>
<td>https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/leetcodesolutions_en_2k</td>
<td>en</td>
</tr>
<tr>
<td>dim/forum_uristov_rf_prompts</td>
<td>https://huggingface.co/datasets/dim/forum_uristov_rf_prompts</td>
<td>Вопросы-ответы с российского юридического форума.</td>
<td>https://xn----dtbrojdkckkfj9k.xn--p1ai/vopros-yuristu?page=560</td>
<td>https://xn----dtbrojdkckkfj9k.xn--p1ai/vopros-yuristu?page=560</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/forum_uristov_rf</td>
<td>ru</td>
</tr>
<tr>
<td>dim/dialogsum_ru</td>
<td>https://huggingface.co/datasets/dim/dialogsum_ru</td>
<td>Саммаризация диалогов на русском языке, перевод dialogsum.</td>
<td>d0rj/dialogsum-ru</td>
<td>https://huggingface.co/datasets/d0rj/dialogsum-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/dialogsum-ru</td>
<td>ru</td>
</tr>
<tr>
<td>dim/huggingartists_prompts</td>
<td>https://huggingface.co/datasets/dim/huggingartists_prompts</td>
<td>Промпты, которые просят продолжить песню в стиле определенного исполнителя. В данном наборе содержатся почти все исполнители, которых вы можете найти в этой организации https://huggingface.co/huggingartists</td>
<td>https://huggingface.co/huggingartists</td>
<td>https://huggingface.co/huggingartists</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/huggingartists</td>
<td>ru</td>
</tr>
</tbody></table>

### Модели

На данный момент обучаются 3 модели llama2_7b, llama2_13b и llama1_30b.

За графиками их обучения можно следить в прямом эфире https://api.wandb.ai/links/dimweb/7rh0c7iz

### Код обучения

- [общий алгоритм обучения](https://github.com/dmitrymailk/verbalist/blob/master/verbalist/model/src/train.py)
- [формирование датасетов для обучения](https://github.com/dmitrymailk/verbalist/blob/master/verbalist/model/src/dataset.py#L176)

### Оборудование

Все обучение и инференс производится на видеокарте A100, на других видеокартах была обнаружена существенная деградация качества при инференсе, данный аспект требует дополнительного изучения.

- NVIDIA A100-SXM4-40GB
- NVIDIA-SMI 535.54.03
- Driver Version: 535.54.03
- CUDA Version: 12.2
- torch==2.0.1+cu118
