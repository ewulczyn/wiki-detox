from db_utils import query_analytics_store
import pandas as pd
import datetime



def get_blocked_users(keywords):
    
    params = {
        'keywords': '|'.join(keywords)
    }
    
    query = """
    SELECT 
        log_comment as reason,
        log_timestamp as timestamp,
        log_title as user_text
    FROM enwiki.logging 
        WHERE log_type = 'block'
        AND log_comment RLIKE %(keywords)s
    """

    return query_analytics_store(query  , params)



def get_blocked_users_talk_page_comment_meta_data(d_blocked_users, max_comments, namespace):
    """
    Given a Dataframe of blocked users, gets the last max_comments
    namespace comments
    """
    
    def shift_date(mw_timestamp, days):
        ts = datetime.datetime.strptime(str(mw_timestamp), "%Y%m%d%H%M%S")
        ts +=  datetime.timedelta(days=days)
        return ts.strftime("%Y%m%d%H%M%S")


    def get_user_talk_page_comment_meta_data(x):
        i, row = x

        blocked_timestamp = row['timestamp']
        user_text = row['user_text']

        params = {
            'blocked_timestamp':blocked_timestamp,
            'user_text':user_text,
            'max_comments': max_comments,
            'min_date': shift_date(blocked_timestamp, -14), 
            'namespace': namespace
        }

        query = """
            SELECT
                page_id,
                page_namespace,
                page_title,
                rev_comment,
                rev_id,
                rev_minor_edit as minor_edit,
                rev_timestamp as timestamp,
                rev_user as user_id,
                rev_user_text as user_text
            FROM
                enwiki.revision r,
                enwiki.page p 
            WHERE
                r.rev_page = p.page_id
                AND rev_timestamp > %(min_date)s
                AND rev_timestamp < %(blocked_timestamp)s
                AND page_namespace = %(namespace)d
                AND rev_user_text = '%(user_text)s'
            ORDER BY rev_timestamp DESC
            LIMIT %(max_comments)d
        """
        try:
            return query_analytics_store(query % params , {})
        except:
            return pd.DataFrame()
    
    t1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        results = executor.map(get_user_talk_page_comment_meta_data, d_blocked_users.iterrows())
        d_meta = pd.concat(results, axis=0)
    t2 = time.time()
    print((t2-t1))
    print('Rate:', float(d_blocked_users.shape[0]) / ((t2-t1)/60.0), 'rpm')
    d_meta.drop_duplicates(subset='rev_id', inplace = True)
    d_meta.index = d_meta['rev_id']
    return d_meta

def get_talk_page_comment_meta_data(n, min_date, namespace):
    """
    Fetches metadata for n talk page comments from namespace
    from after min_date.
    """

    assert(namespace in [1, 3])
    query = """
    SELECT
        page_id,
        page_namespace,
        page_title,
        rev_comment,
        rev_id,
        rev_minor_edit as minor_edit,
        rev_timestamp as timestamp,
        rev_user as user_id,
        rev_user_text as user_text
    FROM
        enwiki.revision r,
        enwiki.page p 
    WHERE
        r.rev_page = p.page_id
        AND rev_timestamp > %(min_date)d
        AND page_namespace = %(namespace)d
    LIMIT %(num_comments)d
    """

    params = {
        'num_comments':n,
        'min_date':min_date,
        'namespace': namespace,
    }

    d = query_analytics_store(query % params , {})
    d.index = d['rev_id']
    return d