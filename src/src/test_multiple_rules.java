//import com.alibaba.fastjson.JSONArray;
//import main.DBConn;
//import main.Rewriter;
//import main.Utils;
//import org.apache.calcite.plan.RelOptRule;
//import org.apache.calcite.rel.rules.CoreRules;
//import org.apache.calcite.rel.rules.PruneEmptyRules;
//
//import java.io.*;
//import java.util.Arrays;
//import java.util.List;
//
//public class test_multiple_rules {
//    public static void main(String[] args) throws Exception{
//        String path = System.getProperty("user.dir");
//        JSONArray schemaJson = Utils.readJsonFile(path+"/src/main/schema.json");
//        Rewriter rewriter = new Rewriter(schemaJson);
//        String testSql = "select * from customer;";
//        testSql = testSql.replace(";", "");
//        RelOptRule rule = CoreRules.AGGREGATE_REMOVE;
//
//
//        File testFile = new File("C:\\Users\\Filene\\Downloads\\sqlsmith_sqls\\card_pc10000_N1000_tmp");
//        FileReader fileReader = null;
//        BufferedReader br = null;
//        String line = null;
//        fileReader = new FileReader(testFile);
//        br = new BufferedReader(fileReader);
//        line = br.readLine();
//        String line2 = "";
//        List<RelOptRule> l = Arrays.<RelOptRule>asList(
//                CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES,
//                // ,CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN,
//                CoreRules.AGGREGATE_JOIN_TRANSPOSE_EXTENDED,CoreRules.AGGREGATE_PROJECT_MERGE,CoreRules.AGGREGATE_ANY_PULL_UP_CONSTANTS,CoreRules.AGGREGATE_UNION_AGGREGATE
//                // CoreRules.FILTER_AGGREGATE_TRANSPOSE,CoreRules.FILTER_CORRELATE,CoreRules.FILTER_INTO_JOIN,CoreRules.JOIN_CONDITION_PUSH,CoreRules.FILTER_MERGE,CoreRules.FILTER_MULTI_JOIN_MERGE, CoreRules.FILTER_PROJECT_TRANSPOSE,CoreRules.FILTER_SET_OP_TRANSPOSE,CoreRules.FILTER_TABLE_FUNCTION_TRANSPOSE,CoreRules.FILTER_SCAN,CoreRules.FILTER_REDUCE_EXPRESSIONS,CoreRules.PROJECT_REDUCE_EXPRESSIONS,PruneEmptyRules.FILTER_INSTANCE,
//                // CoreRules.JOIN_EXTRACT_FILTER,CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE,CoreRules.JOIN_PROJECT_LEFT_TRANSPOSE,CoreRules.JOIN_PROJECT_RIGHT_TRANSPOSE,CoreRules.JOIN_LEFT_UNION_TRANSPOSE,CoreRules.JOIN_RIGHT_UNION_TRANSPOSE,CoreRules.SEMI_JOIN_REMOVE,CoreRules.JOIN_REDUCE_EXPRESSIONS,PruneEmptyRules.JOIN_LEFT_INSTANCE,PruneEmptyRules.JOIN_RIGHT_INSTANCE
//                // CoreRules.UNION_MERGE,CoreRules.UNION_REMOVE,CoreRules.UNION_TO_DISTINCT,CoreRules.UNION_PULL_UP_CONSTANTS,PruneEmptyRules.UNION_INSTANCE,PruneEmptyRules.INTERSECT_INSTANCE,PruneEmptyRules.MINUS_INSTANCE
//                );
//        while(line != null) {
//            line2 = line2 + " " + line;
//            // Notice: the following statement is necessary.
//            line = br.readLine();
//        }
//        fileReader.close();
//        String[] sqls = line2.split(";");
//        String bugs = "";
//        int count_sql = 0, L_SQL = Integer.parseInt(args[0]), R_SQL = Integer.parseInt(args[1]);
//        // FROM L_SQL TO R_SQL;
//        boolean flag[] = new boolean[1005];
//        for(int i = 1; i <= 1000; ++ i)
//            flag[i] = true;
//        int failed = 0;
//        for(RelOptRule rule_agg: l){
//            count_sql = 0;
//            failed = 0;
//            System.out.println(rule_agg.toString());
//            for(String sql : sqls){
//                count_sql ++;
//                if(count_sql > R_SQL) break;
//                if(count_sql >= L_SQL) {
//                    // System.out.println(sqls.length);
//                    try {
//                        if (flag[count_sql] && !test_single_rule.SingleRuleCheck(db, rule_agg, sql, rewriter, card_true, spes_true, not_applied)) {
//                            System.out.println(sql + ";" + rule_agg.toString());
//                            String rewritesql = test_single_rule.SingleRuleApply(rule, testSql, rewriter);
//                            bugs = bugs + sql + ";" + rewritesql+";" + rule_agg.toString() + "\n";
//                        }
//                        if(!flag[count_sql])failed++;
//                    } catch (Exception e) {
//                        System.out.println("our bug~");
//                        flag[count_sql] = false;
//                        failed ++;
//                    }
//                }
//            }
//            bugs = bugs + rule_agg.toString() + "\ncard_true = " + card_true.getValue() + " spes_true = " + spes_true.getValue() + " not applied = " + not_applied.getValue() + " failed = " + failed + "\n";
//            System.out.println(rule_agg.toString() + "\ncard_true = " + card_true.getValue() + " spes_true = " + spes_true.getValue() + " not applied = " + not_applied.getValue() + " failed = " + failed);
//            card_true.setValue(0);
//            spes_true.setValue(0);
//            not_applied.setValue(0);
//        }
//        System.out.println("~~~~~~~~~~~~~~~~");
//        System.out.println(bugs);
//        File file = new File("rule_output/card_pc10000_N1000_tmp_calcite_rule_bug_"+L_SQL+"_"+R_SQL+".txt");
//        FileOutputStream outputStream = null;
//        try {
//            outputStream = new FileOutputStream(file, true);
//            byte[] buf = bugs.getBytes();
//            outputStream.write(buf, 0 , buf.length);
//        } catch (Exception e) {
//            e.printStackTrace();
//        } finally {
//            try {
//                outputStream.close();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
//    }
//}
