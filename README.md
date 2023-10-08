# Verbalist (буквоед) - русскоязычный ассистент.

Проект во многом вдохновленный [Saiga](https://huggingface.co/IlyaGusev/saiga2_7b_lora).

Мною были собраны все самые качественные датасеты с [huggingface.datasets](https://huggingface.co/datasets), а также собраны дополнительно с тех сайтов, которые я посчитал весьма полезными для создания аналога ChatGPT. Лицензии у всех датасетов отличаются, какие-то по типу [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) были созданы специально для обучения подобных моделей, какие-то являются прямой выгрузкой диалогов с ChatGPT ([RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)).

Смысл данного репозитория состоит в систематизации и стандартизации уже имеющихся датасетов, добавлении новых. А также тренировке моделей на этих данных.

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
<th title="Field #8">amount_examples</th>
<th title="Field #9">mean_llama_tokens</th>
<th title="Field #10">std</th>
<th title="Field #11">min_llama_tokens</th>
<th title="Field #12">25%</th>
<th title="Field #13">50%</th>
<th title="Field #14">75%</th>
<th title="Field #15">max_llama_tokens</th>
</tr></thead>
<tbody><tr>
<td>dim/oasst_en</td>
<td>https://huggingface.co/datasets/dim/oasst_en</td>
<td>OpenAssistant Conversations Dataset на английском языке, который был вручную отфильтрован мной. В исходном датасете около 30% диалогов оказались не корректными. Иногда пользователь, играющий роль ассистента, использовал грубый тон в общении с пользователем, иногда люди просто отвечали &quot;не знаю&quot; на вопросы, и некоторые из вопросов были недостаточно научными или слишком краткими. Вы можете ознакомиться с этой разметкой по следующей ссылке: https://docs.google.com/spreadsheets/d/117t5-Tr-dxdODpyFBkBg5R8GklYBlsvBfeDyjqwz2pA/edit?usp=sharing</td>
<td>2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/oasst</td>
<td>en</td>
<td align="right">2289</td>
<td align="right">468.6788991</td>
<td align="right">295.0864391</td>
<td align="right">17</td>
<td align="right">264</td>
<td align="right">410</td>
<td align="right">618</td>
<td align="right">2332</td>
</tr>
<tr>
<td>dim/oasst_ru</td>
<td>https://huggingface.co/datasets/dim/oasst_ru</td>
<td>OpenAssistant Conversations Dataset на русском языке, который был вручную отфильтрован мной. В исходном датасете около 30% диалогов оказались не корректными. Иногда пользователь, играющий роль ассистента, использовал грубый тон в общении с пользователем, иногда люди просто отвечали &quot;не знаю&quot; на вопросы, и некоторые из вопросов были недостаточно научными или слишком краткими. Вы можете ознакомиться с этой разметкой по следующей ссылке: https://docs.google.com/spreadsheets/d/1uiOnqxiytuxrB6u6q2pMSdnMfqjT3arfg8DlT-OWlb0/edit?usp=sharing</td>
<td>2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/2023-04-12_oasst_ready.messages.jsonl.gz</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/oasst</td>
<td>ru</td>
<td align="right">2220</td>
<td align="right">589.6112613</td>
<td align="right">479.835392</td>
<td align="right">7</td>
<td align="right">278</td>
<td align="right">465</td>
<td align="right">763.5</td>
<td align="right">5028</td>
</tr>
<tr>
<td>dim/lima</td>
<td>https://huggingface.co/datasets/dim/lima</td>
<td>Данный датасет включает в себя 1000 высококачественных обучающих примеров на английском языке. Он собран из различных источников, включая Stack Exchange (STEM), Stack Exchange (Other), wikiHow, Pushshift r/WritingPrompts, Natural Instructions, а также уникальные инструкции, созданные авторами статей. Более подробную информацию о датасете можно найти в [соответствующей статье](https://arxiv.org/pdf/2305.11206.pdf).</td>
<td>GAIR/lima</td>
<td>https://huggingface.co/datasets/GAIR/lima</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/lima</td>
<td>en</td>
<td align="right">1030</td>
<td align="right">712.9456311</td>
<td align="right">671.179319</td>
<td align="right">29</td>
<td align="right">312.75</td>
<td align="right">488.5</td>
<td align="right">825</td>
<td align="right">3920</td>
</tr>
<tr>
<td>dim/logic_tasks_ru</td>
<td>https://huggingface.co/datasets/dim/logic_tasks_ru</td>
<td>Данный набор задач по логике для детей взят с веб-сайта https://www.potehechas.ru/zadachi/zadachi.shtml.</td>
<td>Логические задачи - Логика и нестандартное мышление</td>
<td>https://www.potehechas.ru/zadachi/zadachi.shtml</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/logic_tasks_ru</td>
<td>ru</td>
<td align="right">86</td>
<td align="right">193.0697674</td>
<td align="right">76.69048422</td>
<td align="right">58</td>
<td align="right">133.75</td>
<td align="right">185</td>
<td align="right">243.5</td>
<td align="right">432</td>
</tr>
<tr>
<td>dim/wikihow_en</td>
<td>https://huggingface.co/datasets/dim/wikihow_en</td>
<td>Данный датасет содержит англоязычные статьи, извлеченные с веб-сайта Wikihow.</td>
<td>0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/wiki_how</td>
<td>en</td>
<td align="right">1995</td>
<td align="right">2037.86416</td>
<td align="right">870.1910713</td>
<td align="right">265</td>
<td align="right">1463</td>
<td align="right">1913</td>
<td align="right">2461.5</td>
<td align="right">8988</td>
</tr>
<tr>
<td>dim/wikihow_ru</td>
<td>https://huggingface.co/datasets/dim/wikihow_ru</td>
<td>Данный датасет включает в себя русскоязычные статьи, полученные с веб-сайта Wikihow.</td>
<td>0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/wiki_how</td>
<td>ru</td>
<td align="right">2058</td>
<td align="right">2498.119534</td>
<td align="right">1587.851549</td>
<td align="right">139</td>
<td align="right">1236.25</td>
<td align="right">2264</td>
<td align="right">3421.75</td>
<td align="right">10217</td>
</tr>
<tr>
<td>dim/essayforum_writing_prompts_6k</td>
<td>https://huggingface.co/datasets/dim/essayforum_writing_prompts_6k</td>
<td>Данный датасет включает в себя запросы на помощь с написанием небольших эссе, размещенные на данном сайте. Ответы в датасете предоставлены исключительно главным администратором сайта. Его ответы были отобраны, поскольку чаще всего они являются наиболее качественными и вдумчивыми.</td>
<td>EssayForum</td>
<td>https://essayforum.com/writing/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/essayforum</td>
<td>en</td>
<td align="right">6361</td>
<td align="right">783.1760729</td>
<td align="right">285.4314176</td>
<td align="right">258</td>
<td align="right">629</td>
<td align="right">742</td>
<td align="right">879</td>
<td align="right">4966</td>
</tr>
<tr>
<td>dim/sharegpt_short_ru</td>
<td>https://huggingface.co/datasets/dim/sharegpt_short_ru</td>
<td>Очищенная версия русская версия sharegpt. Я попытался вырезать из текста все промпты, где модель извиняется что что-то не может сделать, что она не имеет доступа в интернет. Диалоги, которые противоречат морали модели я просто исключил. Постарался убрать упоминания о том что она модель AI, так как за ролеплейные характеристики отвечают другие датасеты.</td>
<td>RyokoAI/ShareGPT52K</td>
<td>https://huggingface.co/datasets/RyokoAI/ShareGPT52K</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/sharegpt</td>
<td>ru</td>
<td align="right">253</td>
<td align="right">706.6521739</td>
<td align="right">494.7437584</td>
<td align="right">13</td>
<td align="right">310</td>
<td align="right">628</td>
<td align="right">1078</td>
<td align="right">1861</td>
</tr>
<tr>
<td>dim/openreview_prompts_65</td>
<td>https://huggingface.co/datasets/dim/openreview_prompts_65</td>
<td>Датасет рецензий на реальные научные статьи с сайта openreview. Вышло на самом деле не так много, так как многие статьи не выложенны на arxiv или просто не имеют рецензий. Плюс я собрал только малую часть данного сайта, а не все что там было. </td>
<td>https://openreview.net/</td>
<td>https://openreview.net/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/openreview</td>
<td>en</td>
<td align="right">150</td>
<td align="right">13531.51333</td>
<td align="right">6966.623686</td>
<td align="right">4893</td>
<td align="right">8279</td>
<td align="right">12648.5</td>
<td align="right">15833.5</td>
<td align="right">41494</td>
</tr>
<tr>
<td>dim/roleplay_instruct_v2_final</td>
<td>https://huggingface.co/datasets/dim/roleplay_instruct_v2_final</td>
<td>Датасет ролеплея от GPT-4 на различных персонажей на английском языке.</td>
<td>roleplay-instruct-v2-final</td>
<td>https://github.com/teknium1/GPTeacher</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/gpt_roleplay_realm</td>
<td>en</td>
<td align="right">7188</td>
<td align="right">155.1413467</td>
<td align="right">97.71215667</td>
<td align="right">14</td>
<td align="right">88</td>
<td align="right">125</td>
<td align="right">192</td>
<td align="right">1291</td>
</tr>
<tr>
<td>dim/kinomania_scripts</td>
<td>https://huggingface.co/datasets/dim/kinomania_scripts</td>
<td>Небольшой датасет, который содержит в себе сценарии фильмов целиком и их краткое содержание</td>
<td>https://www.kinomania.ru/scripts</td>
<td>https://www.kinomania.ru/scripts</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/kinomania_scripts</td>
<td>ru\en</td>
<td align="right">27</td>
<td align="right">2603.407407</td>
<td align="right">510.375447</td>
<td align="right">1887</td>
<td align="right">2175</td>
<td align="right">2370</td>
<td align="right">3069</td>
<td align="right">3616</td>
</tr>
<tr>
<td>dim/bugurt_thread_prompts</td>
<td>https://huggingface.co/datasets/dim/bugurt_thread_prompts</td>
<td>Небольшой набор размеченных бугуртов вместе с моим другом, для того чтобы модель научилась писать бугурты на конкретную ситуацию. Собраны из телеграм паблика БУГУРТ ТРЕД(https://t.me/bugurtthread)</td>
<td>https://t.me/bugurtthread</td>
<td>https://t.me/bugurtthread</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/bugurt_thread</td>
<td>ru</td>
<td align="right">223</td>
<td align="right">334.4529148</td>
<td align="right">271.2557988</td>
<td align="right">48</td>
<td align="right">148.5</td>
<td align="right">254</td>
<td align="right">434.5</td>
<td align="right">1645</td>
</tr>
<tr>
<td>dim/russian_lyrics_prompts</td>
<td>https://huggingface.co/datasets/dim/russian_lyrics_prompts</td>
<td>Небольшой датасет промптов собранный мною из различных учебников по стихосложению, чтобы модель научилась писать стихи, используя необходимый литературный прием на конкретную тему.</td>
<td>Учебник стихосложения</td>
<td>https://stihi.ru/uchebnik/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/russian_lyrics_prompts</td>
<td>ru</td>
<td align="right">43</td>
<td align="right">106.1395349</td>
<td align="right">71.00220701</td>
<td align="right">45</td>
<td align="right">71</td>
<td align="right">83</td>
<td align="right">96.5</td>
<td align="right">411</td>
</tr>
<tr>
<td>dim/ru_instruct_gpt4</td>
<td>https://huggingface.co/datasets/dim/ru_instruct_gpt4</td>
<td>Датасет каких-то инструкций на русском сгенерированных GPT-4</td>
<td>lksy/ru_instruct_gpt4</td>
<td>https://huggingface.co/datasets/lksy/ru_instruct_gpt4</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ru_instruct_gpt4</td>
<td>ru</td>
<td align="right">14222</td>
<td align="right">259.2173393</td>
<td align="right">237.9433891</td>
<td align="right">16</td>
<td align="right">109</td>
<td align="right">175</td>
<td align="right">271</td>
<td align="right">1374</td>
</tr>
<tr>
<td>dim/gpt_roleplay_realm</td>
<td>https://huggingface.co/datasets/dim/gpt_roleplay_realm</td>
<td>Диалоги выдуманных персонажей при помощи GPT-4, диалоги были сгенерированны при помощи GPT-3.5. Русский и английский.</td>
<td>IlyaGusev/gpt_roleplay_realm</td>
<td>https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/gpt_roleplay_realm</td>
<td>ru\en</td>
<td align="right">8700</td>
<td align="right">504.2424138</td>
<td align="right">117.6228987</td>
<td align="right">180</td>
<td align="right">424</td>
<td align="right">489</td>
<td align="right">569</td>
<td align="right">1207</td>
</tr>
<tr>
<td>dim/ultrachat_ru</td>
<td>https://huggingface.co/datasets/dim/ultrachat_ru</td>
<td>Какой-то рандомный датасет диалогов от chatgpt, который я нашел на huggingface. Из текста диалогов были вырезаны шаблонные фразы по типу: &quot;я не могу выполнить&quot;, &quot;как языковая модель&quot; и тд. Потому что обычно после этого следовало вменяемое решение задачи.</td>
<td>kaleinaNyan/UltraChat_ru</td>
<td>https://huggingface.co/datasets/kaleinaNyan/UltraChat_ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ultrachat_ru</td>
<td>ru</td>
<td align="right">500</td>
<td align="right">1781.782</td>
<td align="right">901.1212735</td>
<td align="right">267</td>
<td align="right">1113.25</td>
<td align="right">1648</td>
<td align="right">2250.25</td>
<td align="right">7303</td>
</tr>
<tr>
<td>dim/scitldr</td>
<td>https://huggingface.co/datasets/dim/scitldr</td>
<td>Саммаризация научных статей на английском языке, выполненная экспертами.</td>
<td>allenai/scitldr</td>
<td>https://huggingface.co/datasets/allenai/scitldr</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/scitldr</td>
<td>en</td>
<td align="right">3229</td>
<td align="right">258.748529</td>
<td align="right">71.41209752</td>
<td align="right">60</td>
<td align="right">209</td>
<td align="right">252</td>
<td align="right">303</td>
<td align="right">689</td>
</tr>
<tr>
<td>dim/linux_man_pages_tldr_summarized</td>
<td>https://huggingface.co/datasets/dim/linux_man_pages_tldr_summarized</td>
<td>Саммаризация мануалов для инструментов линукс в удобный набор команд с их кратким описанием.</td>
<td>tmskss/linux-man-pages-tldr-summarized</td>
<td>https://huggingface.co/datasets/tmskss/linux-man-pages-tldr-summarized</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/linux-man-pages-tldr-summarized</td>
<td>en</td>
<td align="right">481</td>
<td align="right">1567.727651</td>
<td align="right">3590.30871</td>
<td align="right">96</td>
<td align="right">405</td>
<td align="right">765</td>
<td align="right">1386</td>
<td align="right">49888</td>
</tr>
<tr>
<td>dim/dolphin_ru_3k</td>
<td>https://huggingface.co/datasets/dim/dolphin_ru_3k</td>
<td>Подвыборка размера 3000 переведенных заданий dolphin. Примеры из оригинального датасета это промпты из FLANv2 и решения при помощи GPT-4 или GPT-3.5.</td>
<td>d0rj/dolphin-ru</td>
<td>https://huggingface.co/datasets/d0rj/dolphin-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/dolphin_ru</td>
<td>ru</td>
<td align="right">3000</td>
<td align="right">556.1133333</td>
<td align="right">650.0962612</td>
<td align="right">19</td>
<td align="right">207</td>
<td align="right">369.5</td>
<td align="right">720.25</td>
<td align="right">6787</td>
</tr>
<tr>
<td>dim/runne_prompts</td>
<td>https://huggingface.co/datasets/dim/runne_prompts</td>
<td>Промпты составленные из датасета RuNNE. Лично я при обучении сотавил промпт следующим образом. Сначала идет текст &quot;Найди все именованные сущности в данном тексте:&quot;, а затем шел сам текст. В качестве выхода модели нужно сгенерировать JSON где содержатся все найденные именованные сущности. К примеру так [{&quot;name&quot;: &quot;PERSON&quot;, &quot;ent&quot;: &quot;Ким Чен Нама&quot;, &quot;pos&quot;: &quot;0 12&quot;}, {&quot;name&quot;: &quot;ORGANIZATION&quot;, &quot;ent&quot;: &quot;Полиция Малайзии&quot;, &quot;pos&quot;: &quot;56 72&quot;}]</td>
<td>iluvvatar/RuNNE</td>
<td>https://huggingface.co/datasets/iluvvatar/RuNNE</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/RuNNE</td>
<td>ru</td>
<td align="right">537</td>
<td align="right">1479.750466</td>
<td align="right">230.0259174</td>
<td align="right">581</td>
<td align="right">1337</td>
<td align="right">1480</td>
<td align="right">1635</td>
<td align="right">1988</td>
</tr>
<tr>
<td>dim/lurk_prompts</td>
<td>https://huggingface.co/datasets/dim/lurk_prompts</td>
<td>Набор определений различных терминов с сайта lurk. Сами промпты были составлены автоматически следующим образом. напиши определение для (ОПРЕДЕЛЕНИЕ) в стиле lurk</td>
<td>averoo/lurk</td>
<td>https://huggingface.co/datasets/averoo/lurk/viewer/default/train?p=2</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/lurk</td>
<td>ru</td>
<td align="right">5671</td>
<td align="right">3450.34262</td>
<td align="right">4147.897824</td>
<td align="right">35</td>
<td align="right">710.5</td>
<td align="right">2010</td>
<td align="right">4593</td>
<td align="right">55098</td>
</tr>
<tr>
<td>dim/panorama_prompts_10k</td>
<td>https://huggingface.co/datasets/dim/panorama_prompts_10k</td>
<td>Набор юмористических заголовков и текстов новостей с сайта панорама.</td>
<td>its5Q/panorama</td>
<td>https://huggingface.co/datasets/its5Q/panorama</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/panorama</td>
<td>ru</td>
<td align="right">11024</td>
<td align="right">516.9588171</td>
<td align="right">191.3774023</td>
<td align="right">36</td>
<td align="right">422</td>
<td align="right">498</td>
<td align="right">585</td>
<td align="right">3496</td>
</tr>
<tr>
<td>dim/resh_edu_short_prompts</td>
<td>https://huggingface.co/datasets/dim/resh_edu_short_prompts</td>
<td>Набор уроков с сайта resh.edu.ru включающих в себя название урока, тему, класс и текст урока с заданиями. </td>
<td>its5Q/resh-edu</td>
<td>https://huggingface.co/datasets/its5Q/resh-edu</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/resh_edu</td>
<td>ru</td>
<td align="right">2106</td>
<td align="right">1431.510921</td>
<td align="right">435.7847102</td>
<td align="right">56</td>
<td align="right">1175.5</td>
<td align="right">1517</td>
<td align="right">1777</td>
<td align="right">2029</td>
</tr>
<tr>
<td>dim/databricks_dolly_15k_ru</td>
<td>https://huggingface.co/datasets/dim/databricks_dolly_15k_ru</td>
<td>Переведенный датасет dolly на русский язык. Включает в себя набор инструкций на обширное количество тематик.</td>
<td>dwarf2/databricks-dolly-15k-ru</td>
<td>https://huggingface.co/dwarf2/databricks-dolly-15k-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/databricks_dolly_15k_ru</td>
<td>ru</td>
<td align="right">14914</td>
<td align="right">305.4638595</td>
<td align="right">405.874049</td>
<td align="right">8</td>
<td align="right">87</td>
<td align="right">182</td>
<td align="right">370</td>
<td align="right">9268</td>
</tr>
<tr>
<td>dim/databricks_dolly_15k_en</td>
<td>https://huggingface.co/datasets/dim/databricks_dolly_15k_en</td>
<td>databricks-dolly-15k — это набор данных с открытым исходным кодом, содержащий записи о выполнении инструкций, созданные тысячами сотрудников Databricks в нескольких поведенческих категориях, изложенных в документе InstructGPT, включая мозговой штурм, классификацию, закрытый контроль качества, генерацию, извлечение информации, открытый контроль качества и обобщение.</td>
<td>databricks/databricks-dolly-15k</td>
<td>https://huggingface.co/datasets/databricks/databricks-dolly-15k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/databricks_dolly_15k_en</td>
<td>en</td>
<td align="right">15011</td>
<td align="right">204.7264006</td>
<td align="right">302.5539423</td>
<td align="right">6</td>
<td align="right">57</td>
<td align="right">119</td>
<td align="right">242</td>
<td align="right">8883</td>
</tr>
<tr>
<td>dim/grammarly_coedit</td>
<td>https://huggingface.co/datasets/dim/grammarly_coedit</td>
<td>Набор промптов, которые просят исправить грамматические, стилистические ошибки на английском.</td>
<td>grammarly/coedit</td>
<td>https://huggingface.co/datasets/grammarly/coedit</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/grammarly_coedit</td>
<td>en</td>
<td align="right">82466</td>
<td align="right">53.7128271</td>
<td align="right">26.73822864</td>
<td align="right">10</td>
<td align="right">35</td>
<td align="right">46</td>
<td align="right">64</td>
<td align="right">694</td>
</tr>
<tr>
<td>dim/kinopoisk_prompts</td>
<td>https://huggingface.co/datasets/dim/kinopoisk_prompts</td>
<td>Отзывы с кинопоиска на топ 250 фильмов. В промптах я прошу написать хороший, плохой или нейтральный отзыв на определенный фильм.</td>
<td>blinoff/kinopoisk</td>
<td>https://huggingface.co/datasets/blinoff/kinopoisk</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/kinopoisk</td>
<td>ru</td>
<td align="right">36591</td>
<td align="right">875.0955973</td>
<td align="right">565.3212035</td>
<td align="right">48</td>
<td align="right">484</td>
<td align="right">733</td>
<td align="right">1117</td>
<td align="right">8628</td>
</tr>
<tr>
<td>dim/medical_qa_ru_prompts</td>
<td>https://huggingface.co/datasets/dim/medical_qa_ru_prompts</td>
<td>Какие-то вопросы и ответы с какого-то медицинского форума. В данной версии датасета только первый ответ из оригинала.</td>
<td>blinoff/medical_qa_ru_data</td>
<td>https://huggingface.co/datasets/blinoff/medical_qa_ru_data</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/medical_qa_ru_data</td>
<td>ru</td>
<td align="right">80101</td>
<td align="right">206.710528</td>
<td align="right">175.4343973</td>
<td align="right">12</td>
<td align="right">106</td>
<td align="right">161</td>
<td align="right">247</td>
<td align="right">5062</td>
</tr>
<tr>
<td>dim/joke_explaination_prompts</td>
<td>https://huggingface.co/datasets/dim/joke_explaination_prompts</td>
<td>Объяснение шуток на английском. От изначального датасета отличается тем, что я убрал последнее предложение из объяснения, так как оно ссылается на видео на сайте.</td>
<td>theblackcat102/joke_explaination</td>
<td>https://huggingface.co/datasets/theblackcat102/joke_explaination</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/joke_explaination</td>
<td>en</td>
<td align="right">364</td>
<td align="right">143.5741758</td>
<td align="right">68.90275411</td>
<td align="right">21</td>
<td align="right">99</td>
<td align="right">137.5</td>
<td align="right">189.25</td>
<td align="right">334</td>
</tr>
<tr>
<td>dim/oa_stackexchange_200k</td>
<td>https://huggingface.co/datasets/dim/oa_stackexchange_200k</td>
<td>Вопросы-ответы со stackexchange. Оригинальный датасет был составлен следующим образом: были выбраны только темы с принятым ответом, для которых длина вопроса и ответа составляет менее 1000 символов. Другие ответы, вопросы без принятых ответов или длинные записи были удалены. Так как оригинальный датасет слишком большой, я рандомно выбрал 200k семплов.</td>
<td>donfu/oa-stackexchange</td>
<td>https://huggingface.co/datasets/donfu/oa-stackexchange</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/oa_stackexchange</td>
<td>en</td>
<td align="right">200000</td>
<td align="right">276.29862</td>
<td align="right">112.5004436</td>
<td align="right">22</td>
<td align="right">194</td>
<td align="right">265</td>
<td align="right">345</td>
<td align="right">1226</td>
</tr>
<tr>
<td>dim/scale_helpful_no_math</td>
<td>https://huggingface.co/datasets/dim/scale_helpful_no_math</td>
<td>Какой-то набор диалогов с вопросами-ответами на английском, происхождение неизвестно.</td>
<td>HuggingFaceH4/scale_helpful_no_math</td>
<td>https://huggingface.co/datasets/HuggingFaceH4/scale_helpful_no_math/viewer/default/train_rm</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/scale_helpful_no_math</td>
<td>en</td>
<td align="right">17095</td>
<td align="right">1235.302603</td>
<td align="right">838.1097885</td>
<td align="right">53</td>
<td align="right">663</td>
<td align="right">1063</td>
<td align="right">1617</td>
<td align="right">34480</td>
</tr>
<tr>
<td>dim/law_stackexchange_prompts</td>
<td>https://huggingface.co/datasets/dim/law_stackexchange_prompts</td>
<td>Вопросы про закон на английском языке со StackExchange. Оригинальный датасет был преобразован в markdown.</td>
<td>ymoslem/Law-StackExchange</td>
<td>https://huggingface.co/datasets/ymoslem/Law-StackExchange</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/law_stackexchange</td>
<td>en</td>
<td align="right">24343</td>
<td align="right">689.1184324</td>
<td align="right">565.0316906</td>
<td align="right">43</td>
<td align="right">354</td>
<td align="right">540</td>
<td align="right">836</td>
<td align="right">8969</td>
</tr>
<tr>
<td>dim/ficbook_prompts_best_10k</td>
<td>https://huggingface.co/datasets/dim/ficbook_prompts_best_10k</td>
<td>Топ 10k лучших фанфиков с сайта ficbook.net. Все промпты выглядят следующим образом: напиши фанфик с названием {title} и следующим описанием {description}, с тегами {tags}, Где title это оригинальное название, description оригинальное описание, tags это теги данного произведения.</td>
<td>AlexWortega/FicBook</td>
<td>https://huggingface.co/datasets/AlexWortega/FicBook</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ficbook</td>
<td>ru</td>
<td align="right">10000</td>
<td align="right">1737.8214</td>
<td align="right">402.0748161</td>
<td align="right">166</td>
<td align="right">1716</td>
<td align="right">1950</td>
<td align="right">1950</td>
<td align="right">1952</td>
</tr>
<tr>
<td>dim/azbyka_logic_ru</td>
<td>https://huggingface.co/datasets/dim/azbyka_logic_ru</td>
<td>Небольшой набор детских логических и православных задач, взятых с сайта https://azbyka.ru/deti/logicheskie-i-zanimatelnye-zadachi . Обычно у них почти нет развернутого решения, только ответ. Я пытался расписать решение некоторых задач, но меня хватило только на 35, если кто-то займется подобным буду рад https://docs.google.com/spreadsheets/d/1JRbtppbZCUbV_Eqd0nKbRDQEuPnJIAgJ70cUILEDUI4/edit?usp=sharing . </td>
<td>Логические и занимательные задачи (300 задач)</td>
<td>https://azbyka.ru/deti/logicheskie-i-zanimatelnye-zadachi</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/azbyka_logic_ru</td>
<td>ru</td>
<td align="right">480</td>
<td align="right">77.4375</td>
<td align="right">77.56990416</td>
<td align="right">14</td>
<td align="right">31</td>
<td align="right">50</td>
<td align="right">91</td>
<td align="right">652</td>
</tr>
<tr>
<td>dim/povarenok</td>
<td>https://huggingface.co/datasets/dim/povarenok</td>
<td>46k лучших рецептов с сайта povarenok.ru, содержит текст рецепта, список ингридиентов, название блюда</td>
<td>https://www.povarenok.ru/recipes/</td>
<td>https://www.povarenok.ru/recipes/</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/povarenok</td>
<td>ru</td>
<td align="right">46500</td>
<td align="right">488.9118495</td>
<td align="right">344.8563249</td>
<td align="right">31</td>
<td align="right">281</td>
<td align="right">440</td>
<td align="right">632</td>
<td align="right">5542</td>
</tr>
<tr>
<td>dim/AO3_fandom_chatbot_1to1</td>
<td>https://huggingface.co/datasets/dim/AO3_fandom_chatbot_1to1</td>
<td>Какой-то набор ролеплейных диалогов с описанием персонажей и их отыгрышем. Происхождение неизвестно.</td>
<td>ebony59/AO3_fandom_chatbot_1to1</td>
<td>https://huggingface.co/datasets/ebony59/AO3_fandom_chatbot_1to1</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/AO3_fandom_chatbot_1to1</td>
<td>en</td>
<td align="right">614</td>
<td align="right">493.7166124</td>
<td align="right">226.3885365</td>
<td align="right">129</td>
<td align="right">328.25</td>
<td align="right">432.5</td>
<td align="right">611.75</td>
<td align="right">1272</td>
</tr>
<tr>
<td>dim/habr_prompts_5k</td>
<td>https://huggingface.co/datasets/dim/habr_prompts_5k</td>
<td>Статьи с хабра. Датасет был составлен с помощью chatgpt, chatgpt преобразовывал заголовки таким образом чтобы они звучали как вопросы от пользователя, в качестве таргета выступала сама статья.</td>
<td>IlyaGusev/habr</td>
<td>https://huggingface.co/datasets/IlyaGusev/habr</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/habr</td>
<td>ru</td>
<td align="right">5000</td>
<td align="right">1732.892</td>
<td align="right">454.8418369</td>
<td align="right">19</td>
<td align="right">1920.75</td>
<td align="right">1950</td>
<td align="right">1951</td>
<td align="right">1952</td>
</tr>
<tr>
<td>dim/what_where_when_50k</td>
<td>https://huggingface.co/datasets/dim/what_where_when_50k</td>
<td>50k вопросов с решениями с сайта что где когда. В качестве промпта выступает вопрос, в качестве ответа конкатенация объяснения и краткого ответа. Все вопросы-ответы вы можете найти по этой ссылке https://huggingface.co/datasets/dim/what_where_when_ru</td>
<td>https://db.chgk.info</td>
<td>https://db.chgk.info</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/what_where_when</td>
<td>ru</td>
<td align="right">50000</td>
<td align="right">169.1862</td>
<td align="right">68.91119898</td>
<td align="right">18</td>
<td align="right">122</td>
<td align="right">158</td>
<td align="right">202</td>
<td align="right">1167</td>
</tr>
<tr>
<td>dim/competition_math</td>
<td>https://huggingface.co/datasets/dim/competition_math</td>
<td>Датасет олимпиадной математики на английском. The Mathematics Aptitude Test of Heuristics (MATH) dataset.</td>
<td>competition_math</td>
<td>https://huggingface.co/datasets/competition_math</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/competition_math</td>
<td>en</td>
<td align="right">7500</td>
<td align="right">317.5254667</td>
<td align="right">267.8583731</td>
<td align="right">34</td>
<td align="right">147</td>
<td align="right">234</td>
<td align="right">393</td>
<td align="right">3029</td>
</tr>
<tr>
<td>dim/sharegpt_short_en_30k</td>
<td>https://huggingface.co/datasets/dim/sharegpt_short_en_30k</td>
<td>Короткие диалоги на английском из sharegpt</td>
<td>RyokoAI/ShareGPT52K</td>
<td>https://huggingface.co/datasets/RyokoAI/ShareGPT52K</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/sharegpt</td>
<td>en</td>
<td align="right">29597</td>
<td align="right">749.3149981</td>
<td align="right">516.3702473</td>
<td align="right">3</td>
<td align="right">336</td>
<td align="right">630</td>
<td align="right">1095</td>
<td align="right">2021</td>
</tr>
<tr>
<td>dim/ru_turbo_alpaca_evol_instruct</td>
<td>https://huggingface.co/datasets/dim/ru_turbo_alpaca_evol_instruct</td>
<td>Набор инструкций различной тематики на русском языке, сгенерированных при помощи chatgpt.</td>
<td>IlyaGusev/ru_turbo_alpaca_evol_instruct</td>
<td>https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca_evol_instruct</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ru_turbo_alpaca_evol_instruct</td>
<td>ru</td>
<td align="right">47793</td>
<td align="right">453.0887996</td>
<td align="right">289.5498356</td>
<td align="right">17</td>
<td align="right">221</td>
<td align="right">430</td>
<td align="right">623</td>
<td align="right">4647</td>
</tr>
<tr>
<td>dim/ru_turbo_saiga</td>
<td>https://huggingface.co/datasets/dim/ru_turbo_saiga</td>
<td>Набор инструкций различной тематики на русском языке, сгенерированных при помощи chatgpt.</td>
<td>IlyaGusev/ru_turbo_saiga</td>
<td>https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/ru_turbo_saiga</td>
<td>ru</td>
<td align="right">37699</td>
<td align="right">412.7508687</td>
<td align="right">113.346917</td>
<td align="right">87</td>
<td align="right">339</td>
<td align="right">398</td>
<td align="right">466</td>
<td align="right">1427</td>
</tr>
<tr>
<td>dim/bugurt_completion_prompts</td>
<td>https://huggingface.co/datasets/dim/bugurt_completion_prompts</td>
<td>Обрезанные бугурты, где в качестве промпта используется строка вида - продолжи бугурт: первая строчка бугурта </td>
<td>https://t.me/bugurtthread</td>
<td>https://t.me/bugurtthread</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/bugurt_thread</td>
<td>ru</td>
<td align="right">5000</td>
<td align="right">280.2466</td>
<td align="right">320.4353681</td>
<td align="right">32</td>
<td align="right">111</td>
<td align="right">178</td>
<td align="right">331</td>
<td align="right">11333</td>
</tr>
<tr>
<td>dim/tldr_17_50k</td>
<td>https://huggingface.co/datasets/dim/tldr_17_50k</td>
<td>Очень вольная абстрактная саммаризация постов с реддита в одну строчку</td>
<td>webis/tldr-17</td>
<td>https://huggingface.co/datasets/webis/tldr-17</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/tldr_17</td>
<td>en</td>
<td align="right">50000</td>
<td align="right">421.12752</td>
<td align="right">403.346214</td>
<td align="right">10</td>
<td align="right">177</td>
<td align="right">303</td>
<td align="right">525</td>
<td align="right">9592</td>
</tr>
<tr>
<td>dim/grade_school_math_instructions</td>
<td>https://huggingface.co/datasets/dim/grade_school_math_instructions</td>
<td>OpenAI&#39;s grade-school-math датасет преобразованный в промпты.</td>
<td>qwedsacf/grade-school-math-instructions</td>
<td>https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/grade-school-math-instructions</td>
<td>en</td>
<td align="right">8792</td>
<td align="right">171.6310282</td>
<td align="right">63.09232668</td>
<td align="right">50</td>
<td align="right">124</td>
<td align="right">161</td>
<td align="right">206</td>
<td align="right">511</td>
</tr>
<tr>
<td>dim/tldr_news</td>
<td>https://huggingface.co/datasets/dim/tldr_news</td>
<td>Хедлайны и текст новостей на различную тематику. </td>
<td>JulesBelveze/tldr_news</td>
<td>https://huggingface.co/datasets/JulesBelveze/tldr_news</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/tldr_news</td>
<td>en</td>
<td align="right">7138</td>
<td align="right">133.1004483</td>
<td align="right">46.48736493</td>
<td align="right">23</td>
<td align="right">100</td>
<td align="right">133</td>
<td align="right">161</td>
<td align="right">476</td>
</tr>
<tr>
<td>dim/grade_school_math_instructions_ru</td>
<td>https://huggingface.co/datasets/dim/grade_school_math_instructions_ru</td>
<td>OpenAI&#39;s grade-school-math датасет переведенный на русский.</td>
<td>d0rj/gsm8k-ru</td>
<td>https://huggingface.co/datasets/d0rj/gsm8k-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/grade_school_math_instructions_ru</td>
<td>ru</td>
<td align="right">7473</td>
<td align="right">259.8321959</td>
<td align="right">100.1229127</td>
<td align="right">78</td>
<td align="right">185</td>
<td align="right">241</td>
<td align="right">314</td>
<td align="right">838</td>
</tr>
<tr>
<td>dim/dialogsum</td>
<td>https://huggingface.co/datasets/dim/dialogsum</td>
<td>Саммаризация диалогов на английском языке, разметка выполнялась вручную.</td>
<td>knkarthick/dialogsum</td>
<td>https://huggingface.co/datasets/knkarthick/dialogsum</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/dialogsum</td>
<td>en</td>
<td align="right">12460</td>
<td align="right">269.6467095</td>
<td align="right">126.285664</td>
<td align="right">75</td>
<td align="right">191</td>
<td align="right">245</td>
<td align="right">327</td>
<td align="right">1725</td>
</tr>
<tr>
<td>dim/HC3_ru</td>
<td>https://huggingface.co/datasets/dim/HC3_ru</td>
<td>Вопросы-ответы с реддита, есть ответы сгенерированные chatgpt и реальные ответы пользователей. Я использовал только реальные ответы пользователей.</td>
<td>d0rj/HC3-ru</td>
<td>https://huggingface.co/datasets/d0rj/HC3-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/HC3_ru</td>
<td>ru</td>
<td align="right">24322</td>
<td align="right">360.5608503</td>
<td align="right">330.2285903</td>
<td align="right">15</td>
<td align="right">168</td>
<td align="right">267</td>
<td align="right">435</td>
<td align="right">10025</td>
</tr>
<tr>
<td>dim/horoscopes_ru_10k</td>
<td>https://huggingface.co/datasets/dim/horoscopes_ru_10k</td>
<td>10k гороскопов, с промптами где я прошу сгенерировать гороском для определенного знака зодиака</td>
<td>dkagramanyan/horoscopes_ru</td>
<td>https://huggingface.co/datasets/dkagramanyan/horoscopes_ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/horoscopes_ru</td>
<td>ru</td>
<td align="right">10000</td>
<td align="right">183.1443</td>
<td align="right">31.62023184</td>
<td align="right">55</td>
<td align="right">159</td>
<td align="right">187</td>
<td align="right">201</td>
<td align="right">464</td>
</tr>
<tr>
<td>dim/yandex_q_200k</td>
<td>https://huggingface.co/datasets/dim/yandex_q_200k</td>
<td>200k рандомно выбранных вопросов-ответов с сайта yandex q.</td>
<td>its5Q/yandex-q</td>
<td>https://huggingface.co/datasets/its5Q/yandex-q</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/yandex_q</td>
<td>ru</td>
<td align="right">200000</td>
<td align="right">304.569005</td>
<td align="right">340.7808288</td>
<td align="right">18</td>
<td align="right">127</td>
<td align="right">202</td>
<td align="right">353</td>
<td align="right">19294</td>
</tr>
<tr>
<td>dim/leetcodesolutions_en_2k</td>
<td>https://huggingface.co/datasets/dim/leetcodesolutions_en_2k</td>
<td>Решения задач с leetcode на разных языках. </td>
<td>TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k</td>
<td>https://huggingface.co/datasets/TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/leetcodesolutions_en_2k</td>
<td>en</td>
<td align="right">2048</td>
<td align="right">740.7441406</td>
<td align="right">253.2493282</td>
<td align="right">297</td>
<td align="right">565</td>
<td align="right">685</td>
<td align="right">857</td>
<td align="right">1960</td>
</tr>
<tr>
<td>dim/forum_uristov_rf_prompts</td>
<td>https://huggingface.co/datasets/dim/forum_uristov_rf_prompts</td>
<td>Вопросы-ответы с российского юридического форума.</td>
<td>https://xn----dtbrojdkckkfj9k.xn--p1ai/vopros-yuristu?page=560</td>
<td>https://xn----dtbrojdkckkfj9k.xn--p1ai/vopros-yuristu?page=560</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/forum_uristov_rf</td>
<td>ru</td>
<td align="right">1849</td>
<td align="right">321.0540833</td>
<td align="right">429.58896</td>
<td align="right">31</td>
<td align="right">134</td>
<td align="right">210</td>
<td align="right">349</td>
<td align="right">6470</td>
</tr>
<tr>
<td>dim/dialogsum_ru</td>
<td>https://huggingface.co/datasets/dim/dialogsum_ru</td>
<td>Саммаризация диалогов на русском языке, перевод dialogsum.</td>
<td>d0rj/dialogsum-ru</td>
<td>https://huggingface.co/datasets/d0rj/dialogsum-ru</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/dialogsum-ru</td>
<td>ru</td>
<td align="right">12460</td>
<td align="right">364.2813804</td>
<td align="right">178.7117754</td>
<td align="right">98</td>
<td align="right">250</td>
<td align="right">329</td>
<td align="right">446</td>
<td align="right">2300</td>
</tr>
<tr>
<td>dim/huggingartists_prompts</td>
<td>https://huggingface.co/datasets/dim/huggingartists_prompts</td>
<td>Промпты, которые просят продолжить песню в стиле определенного исполнителя. В данном наборе содержатся почти все исполнители, которых вы можете найти в этой организации https://huggingface.co/huggingartists</td>
<td>https://huggingface.co/huggingartists</td>
<td>https://huggingface.co/huggingartists</td>
<td>https://github.com/dmitrymailk/verbalist/tree/master/verbalist/datasets/huggingartists</td>
<td>ru</td>
<td align="right">64006</td>
<td align="right">561.6732025</td>
<td align="right">586.18458</td>
<td align="right">28</td>
<td align="right">297</td>
<td align="right">453</td>
<td align="right">720</td>
<td align="right">32949</td>
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

### Дальнейшее развитие

Самое простое, что можно сделать это переводить уже имеющиеся хорошие датасеты с английского на русский при помощи GPT-4.

Более сложное это собирать больше разнообразных данных из различных доменов. Я могу лишь подкинуть идеи для того какие датасеты можно собрать еще.

- решебники по литературе, русскому и другим предметам
- задания со всяких бирж труда
- [краткие пересказы произведений, анализ произведений, сочинения по ним](http://www.litra.ru/shortwork/)
- [туториалы с digital ocean (более 7000)](https://www.digitalocean.com/community/tutorials)
- [туториалы с selectel](https://selectel.ru/blog/tutorials/)
- больше форумов на различные тематики
- [бесплатные эссе с ivypanda essays](https://ivypanda.com/essays/) и дальнейший их перевод на русский
- больше стихов и песен
- [олимпиадные русские задачи](https://math.ru/problems/) их очень сложно собирать, так как большинство их них живут только в PDF или docx. Но их довольно много и они довольно отличаются от олимпиадной математики на английском. Но у меня нет времени этим заниматься.
- фанфики на иностранном языке
- исправить текущие автоматические промпты на более разнообразные, при помощи chatgpt
