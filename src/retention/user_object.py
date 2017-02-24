class User():
    def __init__(self, user_text, df_annotated_user_text_groups, df_annotated_page_title_groups, df_edits_groups, df_user_groups, df_uw_groups):
        self.user_text = user_text
        self.df_activity =  df_edits_groups.get(user_text, None)
        self.df_comments_made =  df_annotated_user_text_groups.get(user_text, None)
        self.df_comments_received = df_annotated_page_title_groups.get(user_text, None)
        self.df_uw = df_uw_groups.get(user_text, None)
        if self.df_comments_received is not None:
            self.df_comments_received = self.df_comments_received.query("ns == 'user' and user_text != page_title")
        self.gender = df_user_groups[user_text]['gender'].iloc[0]
        self.registration_day = df_user_groups[user_text]['registration_day'].iloc[0]
        self.first_edit_day = df_user_groups[user_text]['first_edit_day'].iloc[0]