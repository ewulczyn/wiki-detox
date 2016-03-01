mwdiffs dump2diffs \
/mnt/data/xmldatadumps/public/enwiki/20160113/enwiki-20160113-pages-meta-history*.xml*.bz2 \
--config ./mwdiff.yaml \
--namespaces=3 \
--timeout=60 \
--output=/srv/ellery/talk_diffs \
--verbose \
