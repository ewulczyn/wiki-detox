stat3
screen -r diff
source ~/env/3.4/bin/activate



name=article_talk
day=20160113
namespaces=1



json_dir=/srv/ellery/${name}_diffs_json

mkdir ${json_dir}

nice mwdiffs dump2diffs \
/mnt/data/xmldatadumps/public/enwiki/${day}/enwiki-${day}-pages-meta-history*.xml*.bz2 \
--config ~//mwdiff.yaml \
--namespaces=${namespaces} \
--timeout=60 \
--output=${json_dir} \
--verbose 


tsv_dir=/srv/public-datasets/enwiki/${name}_diffs_tsv

nice python ~/mwdiffs_to_tsv.py \
--path_glob ${json_dir}/enwiki-${day}-pages-meta-history\*.bz2  \
--output_dir ${tsv_dir}