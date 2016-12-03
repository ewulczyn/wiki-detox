stat3
screen -r diff
source ~/env/3.4/bin/activate



name=user_talk
namespaces=3
day=20160113

json_dir=/srv/ellery/${name}_diffs_json
tsv_dir=/srv/public-datasets/enwiki/${name}_diffs_tsv

__rm -rf ${json_dir}
mkdir ${json_dir}
nice mwdiffs dump2diffs \
/mnt/data/xmldatadumps/public/enwiki/${day}/enwiki-${day}-pages-meta-history*.xml*.bz2 \
--config ~//mwdiff.yaml \
--namespaces=${namespaces} \
--timeout=60 \
--output=${json_dir} \
--verbose 


rm -rf ${tsv_dir}
mkdir ${tsv_dir}
nice python ~/mwdiffs_to_tsv.py \
--path_glob ${json_dir}/enwiki-${day}-pages-meta-history\*.bz2  \
--output_dir ${tsv_dir}


localhost
rsync -avzh stat1003.eqiad.wmnet:/srv/public-datasets/enwiki/${name}_diffs_tsv ~/detox/data
rsync -avzh ~/detox/data/${name}_diffs_tsv stat1002.eqiad.wmnet:/home/ellery/detox/data




stat2

ns=user
cd ~/detox/data/${ns}_talk_diffs_tsv
bzip2 -d *.bz2

cd ..

hadoop fs -mkdir /user/ellery/talk_diff_external
hadoop fs -rm -r -f /user/ellery/talk_diff_external/ns=${ns}
hadoop fs -copyFromLocal ${ns}_talk_diffs_tsv /user/ellery/talk_diff_external/ns=${ns}
