class User():
    def __init__(self, user_id, df_comments_from_groups, df_comments_to_groups, df_edits_groups, df_user_groups, df_uw_groups):
        self.user_id = user_id
        self.df_activity =  df_edits_groups.get(user_id, None)
        self.df_comments_from =  df_comments_from_groups.get(user_id, None)
        self.df_comments_to = df_comments_to_groups.get(user_id, None)
        self.df_uw = df_uw_groups.get(user_id, None)
        self.gender = df_user_groups[user_id]['gender'].iloc[0]
        self.first_edit_day = df_user_groups[user_id]['first_edit_day'].iloc[0]