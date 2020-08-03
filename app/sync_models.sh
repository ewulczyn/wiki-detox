GCE:
scp -r  -i ~/.ssh/google_rsa ~/talk_page_abuse/models  ellerywulczyn@104.196.137.229:talk_page_abuse/

Labs:
https://wikitech.wikimedia.org/wiki/Help:Tool_Labs/Web/Kubernetes#python_.28uwsgi_.2B_python3.4.29




scp -i ~/.ssh/private_key -r /Users/ellerywulczyn/detox/app/models/attack_linear_char_oh_pipeline.pkl ewulczyn@dev.toolforge.org:/home/ewulczyn/models/
scp -i ~/.ssh/private_key -r /Users/ellerywulczyn/detox/app/models/aggression_linear_char_oh_pipeline.pkl ewulczyn@dev.toolforge.org:/home/ewulczyn/models/


tool

cp  models/* /data/project/detox/models

become detox


mv  models/* /data/project/detox/www/python/detox/app/models
