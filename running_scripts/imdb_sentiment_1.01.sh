nohup python experiments/imdb_sentiment.py --models roberta electra_small_discriminator > logs/imdb_sentiment/roberta_lr0.005_pr0.5_np_100_ct0.5_batch256_$(date +"%Y-%m-%d-%H-%M-%S").txt 2>&1 &

nohup python experiments/imdb_sentiment.py --models albert bert > logs/imdb_sentiment/albert_lr0.005_pr0.5_np_100_ct0.5_batch256_$(date +"%Y-%m-%d-%H-%M-%S").txt 2>&1 &

nohup python experiments/imdb_sentiment.py --models distilbert t5  > logs/imdb_sentiment/distilbert_lr0.005_pr0.5_np_100_ct0.5_batch256_$(date +"%Y-%m-%d-%H-%M-%S").txt 2>&1 &

# nohup python experiments/imdb_sentiment.py --models deberta  > logs/imdb_sentiment/deberta_lr0.005_pr0.5_np_100_ct0.5_batch256_$(date +"%Y-%m-%d-%H-%M-%S").txt 2>&1 &

# nohup python experiments/imdb_sentiment.py --models deberta_small  > logs/imdb_sentiment/deberta_small_lr0.005_pr0.5_np_100_ct0.5_batch256_$(date +"%Y-%m-%d-%H-%M-%S").txt 2>&1 &

nohup python experiments/imdb_sentiment.py --models bert  > logs/imdb_sentiment/bert_lr0.005_pr0.5_np_100_ct0.5_batch256_$(date +"%Y-%m-%d-%H-%M-%S").txt 2>&1 &