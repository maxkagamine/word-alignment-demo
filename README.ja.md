# Demonstration of AI/neural word alignment of English & Japanese text using mBERT-based machine learning models.<br>日本語と英語の文に対してmBERTベースの機械学習モデルを使用したAI・ニューラル単語アラインメントのデモ｡

![](screenshot.png)

[English ver.](README.md)

私は2つの最先端英日単語アラインメントのツールを実験して比べるための可視化ツールを作成した。両方は[多言語BERT](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/)に基づいているけど問題を別の方法でアプローチする：

- [**WSPAlign**](https://github.com/qiyuw/WSPAlign)はNTTの[初期作品](https://github.com/nttcslab-nlp/word_align)に基づいて、[question-answering](https://medium.com/analytics-vidhya/question-answering-system-with-bert-ebe1130f8def)を使って _question_（"from"テクスト）でマークされた単語に対応する _context_（"to"テクスト）スパンを探して、"from"テクストにあるそれぞれの単語に対して繰り返す。（※このリポジトリが使うのは[モデル](https://huggingface.co/qiyuw/WSPAlign-ft-kftt)だけ。[元のは](https://github.com/qiyuw/WSPAlign.InferEval)FOSSじゃないので推論コードがゼロから作られてきた）

- [**awesome-align**](https://github.com/neulab/awesome-align)はBERTモデルを通じて両方の文を実行して、文脈化された[単語埋め込み](https://is-rajapaksha.medium.com/bert-word-embeddings-deep-dive-32f6214f02bf)をモデルの「経験的に選べた」レイヤー（[論文](https://arxiv.org/pdf/2101.08231)の2.1節を参照）から抜き出して、可能性の高い単語アラインメントを識別するために類似比較を行う。この方法の利点は一度に文全体を処理できること。

両方のモデル（WSPAlignと微調整されたawesome-alignモデル）は京都に関するウィキペディア記事から取って手動でアラインメントされた1,235の日英文ペアで構成される[京都フリー翻訳タスク（KFTT）](https://www.phontron.com/kftt/index-ja.html)コーパスに学習された。（WSPAlignはヒューリスティックスによって自動的にアラインメントされたウィキペディアから色々の言語の文にさらに学習された。[その論文](https://aclanthology.org/2023.acl-long.621.pdf)の2.3節を参照して）

## Thoughts & conclusions　思考と結論

私の最初の仮説は実際の単語埋め込みを使用するawesome-alignの方が¶文字に包まれている単語の翻訳を見つけるように単純にAIを「聞く」ことより良い結果を出すことだったけど、少なくとも私のテストでは、実際はその逆みたいだ。完璧じゃないけど。上に見える通り、WSPAlignのモデルがいくつくの間違いをした：experienceを「練度」じゃなくて「活躍」に合わせるとか、「上がれば」をas good asに合わせるとか（構文の違いにAIは足元をすくわれるようだ）それに敬称「お」のかなり新規な解釈。

相応しい学習データが足りないのは問題かもしれない。私の知ってる限り、日本語と英語のために存在して手動で単語アラインメントされた唯一の対訳コーパスはKFTTだ。[mBERTのトークン化](https://github.com/google-research/bert/blob/master/multilingual.md#tokenization)が日本語を対応する方法が要素という可能性もある。漢字の単語は全て[個々の文字に分割されるんだ](https://qiita.com/tmitani/items/e520e0a085c9e4ee69ed)。東北大学の[bert-japanese](https://github.com/cl-tohoku/bert-japanese)はMeCabを活用する日本語固有のより良いトーケン化を使うから、それに基づいて新しいモデルを微調整すると精度を向上させるのか興味がある。

それにしても、どれくらい正しいかは結構すごいだ。モデルの効果を見るために色々の例文で自分で試してみるとおすすめする。

![](screenshot-hover.avif)

## Running the visualization server　可視化サーバーの実行

```
$ git clone https://github.com/maxkagamine/word-alignment-demo.git
$ cd word-alignment-demo
$ python3 -m venv .venv && . .venv/bin/activate
$ pip install -r requirements.txt
$ ./visualize.py
```

WSPAlignとBERTのモデルは自動的にダウンロードされるけど、awesome-alignの微調整された「model_with_co」と「model_without_co」モデルは[そのreadme](https://github.com/neulab/awesome-align?tab=readme-ov-file#model-performance)のGoogle Driveリンクから自分でダウンロードして、リポジトリのルートにある「models」というフォルダに抽出する必要がある。 

## Related academic papers　関連学術論文

Wu, Q., Nagata, M., & Tsuruoka, Y. (2023). [WSPAlign: Word Alignment Pre-training via Large-Scale Weakly Supervised Span Prediction](https://aclanthology.org/2023.acl-long.621/). In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ (pp. 11084–11099). Association for Computational Linguistics.

Dou, Z.Y., & Neubig, G. (2021). [Word Alignment by Fine-tuning Embeddings on Parallel Corpora](https://arxiv.org/abs/2101.08231). In _Conference of the European Chapter of the Association for Computational Linguistics (EACL)_.

Nagata, M., Chousa, K., & Nishino, M. (2020). [A Supervised Word Alignment Method based on Cross-Language Span Prediction using Multilingual BERT](https://aclanthology.org/2020.emnlp-main.41/). In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ (pp. 555–565). Association for Computational Linguistics.

