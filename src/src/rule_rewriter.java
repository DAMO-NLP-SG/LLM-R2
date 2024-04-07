import com.alibaba.fastjson.JSONArray;
import main.Rewriter;
import main.EquivCheck;
import main.Utils;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.hep.HepMatchOrder;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rel2sql.RelToSqlConverter;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.PruneEmptyRules;
import org.apache.calcite.sql.dialect.PostgresqlSqlDialect;
// import org.apache.calcite.sql.dialect.SqlDialect;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import java.util.Scanner;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.List;


public class rule_rewriter {

  public static HepProgramBuilder getbuilder(RelOptRule rule_instance){
    HepProgramBuilder builder = new HepProgramBuilder();
    builder.addRuleInstance(rule_instance);

        /*List<RelOptRule> t = Arrays.<RelOptRule>asList(CoreRules.AGGREGATE_PROJECT_MERGE,CoreRules.AGGREGATE_VALUES, PruneEmptyRules.AGGREGATE_INSTANCE);
        for(RelOptRule rule:t){
            builder.addRuleInstance(rule);
        }*/
    return builder;
  }

  public static void main(String[] args) throws Exception {
    Scanner scanner = new Scanner(System.in);
    String inputs = scanner.nextLine();
    Gson gson = new Gson();
    Type type = new TypeToken<List<Object>>(){}.getType();
    List<Object> inputList = gson.fromJson(inputs, type);
    String db_id = (String) inputList.get(0);
    String input_sql = (String) inputList.get(1);
    List<String> input_rules = (List<String>) inputList.get(2);

    // IF YOU CHANGE THE MAP rule2ruleset, THEN DON'T FORGET TO CHANGE THE FILE standard.txt IN ORDER!!!
    HashMap<String, RelOptRule> rulename2rule = new HashMap<>();
    rulename2rule.put("AGGREGATE_EXPAND_DISTINCT_AGGREGATES", CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES);
    rulename2rule.put("AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN", CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN);
    rulename2rule.put("AGGREGATE_JOIN_TRANSPOSE_EXTENDED", CoreRules.AGGREGATE_JOIN_TRANSPOSE_EXTENDED);
    rulename2rule.put("AGGREGATE_PROJECT_MERGE", CoreRules.AGGREGATE_PROJECT_MERGE);
    rulename2rule.put("AGGREGATE_ANY_PULL_UP_CONSTANTS", CoreRules.AGGREGATE_ANY_PULL_UP_CONSTANTS);
    rulename2rule.put("AGGREGATE_UNION_AGGREGATE", CoreRules.AGGREGATE_UNION_AGGREGATE);
    rulename2rule.put("AGGREGATE_UNION_TRANSPOSE", CoreRules.AGGREGATE_UNION_TRANSPOSE);
    rulename2rule.put("AGGREGATE_VALUES", CoreRules.AGGREGATE_VALUES);
    rulename2rule.put("AGGREGATE_REMOVE", CoreRules.AGGREGATE_REMOVE);
    rulename2rule.put("AGGREGATE_INSTANCE", PruneEmptyRules.AGGREGATE_INSTANCE);

    rulename2rule.put("FILTER_AGGREGATE_TRANSPOSE", CoreRules.FILTER_AGGREGATE_TRANSPOSE);
    rulename2rule.put("FILTER_CORRELATE", CoreRules.FILTER_CORRELATE);
    rulename2rule.put("FILTER_INTO_JOIN", CoreRules.FILTER_INTO_JOIN);
    rulename2rule.put("JOIN_CONDITION_PUSH", CoreRules.JOIN_CONDITION_PUSH);
    rulename2rule.put("FILTER_MERGE", CoreRules.FILTER_MERGE);
    rulename2rule.put("FILTER_SCAN", CoreRules.FILTER_SCAN);
    rulename2rule.put("FILTER_MULTI_JOIN_MERGE", CoreRules.FILTER_MULTI_JOIN_MERGE);
    rulename2rule.put("FILTER_PROJECT_TRANSPOSE", CoreRules.FILTER_PROJECT_TRANSPOSE);
    rulename2rule.put("FILTER_SET_OP_TRANSPOSE", CoreRules.FILTER_SET_OP_TRANSPOSE);
    rulename2rule.put("FILTER_TABLE_FUNCTION_TRANSPOSE", CoreRules.FILTER_TABLE_FUNCTION_TRANSPOSE);
    rulename2rule.put("FILTER_REDUCE_EXPRESSIONS", CoreRules.FILTER_REDUCE_EXPRESSIONS);
    rulename2rule.put("FILTER_INSTANCE", PruneEmptyRules.FILTER_INSTANCE);

    rulename2rule.put("JOIN_EXTRACT_FILTER", CoreRules.JOIN_EXTRACT_FILTER);
    rulename2rule.put("JOIN_PROJECT_BOTH_TRANSPOSE", CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE);
    rulename2rule.put("JOIN_PROJECT_LEFT_TRANSPOSE", CoreRules.JOIN_PROJECT_LEFT_TRANSPOSE);
    rulename2rule.put("JOIN_PROJECT_RIGHT_TRANSPOSE", CoreRules.JOIN_PROJECT_RIGHT_TRANSPOSE);
    rulename2rule.put("JOIN_LEFT_UNION_TRANSPOSE", CoreRules.JOIN_LEFT_UNION_TRANSPOSE);
    rulename2rule.put("JOIN_RIGHT_UNION_TRANSPOSE", CoreRules.JOIN_RIGHT_UNION_TRANSPOSE);
    rulename2rule.put("SEMI_JOIN_REMOVE", CoreRules.SEMI_JOIN_REMOVE);
    rulename2rule.put("JOIN_REDUCE_EXPRESSIONS", CoreRules.JOIN_REDUCE_EXPRESSIONS);
    rulename2rule.put("JOIN_LEFT_INSTANCE", PruneEmptyRules.JOIN_LEFT_INSTANCE);
    rulename2rule.put("JOIN_RIGHT_INSTANCE", PruneEmptyRules.JOIN_RIGHT_INSTANCE);

    rulename2rule.put("PROJECT_CALC_MERGE", CoreRules.PROJECT_CALC_MERGE);
    rulename2rule.put("PROJECT_CORRELATE_TRANSPOSE", CoreRules.PROJECT_CORRELATE_TRANSPOSE);
    rulename2rule.put("PROJECT_MERGE", CoreRules.PROJECT_MERGE);
    rulename2rule.put("PROJECT_MULTI_JOIN_MERGE", CoreRules.PROJECT_MULTI_JOIN_MERGE);
    rulename2rule.put("PROJECT_REMOVE", CoreRules.PROJECT_REMOVE);
    rulename2rule.put("PROJECT_TO_CALC", CoreRules.PROJECT_TO_CALC);
    rulename2rule.put("PROJECT_SUB_QUERY_TO_CORRELATE", CoreRules.PROJECT_SUB_QUERY_TO_CORRELATE);
    rulename2rule.put("PROJECT_REDUCE_EXPRESSIONS", CoreRules.PROJECT_REDUCE_EXPRESSIONS);
    rulename2rule.put("PROJECT_INSTANCE", PruneEmptyRules.PROJECT_INSTANCE);

    rulename2rule.put("CALC_MERGE", CoreRules.CALC_MERGE);
    rulename2rule.put("CALC_REMOVE", CoreRules.CALC_REMOVE);

    rulename2rule.put("SORT_JOIN_TRANSPOSE", CoreRules.SORT_JOIN_TRANSPOSE);
    rulename2rule.put("SORT_PROJECT_TRANSPOSE", CoreRules.SORT_PROJECT_TRANSPOSE);
    rulename2rule.put("SORT_UNION_TRANSPOSE", CoreRules.SORT_UNION_TRANSPOSE);
    rulename2rule.put("SORT_REMOVE_CONSTANT_KEYS", CoreRules.SORT_REMOVE_CONSTANT_KEYS);
    rulename2rule.put("SORT_REMOVE", CoreRules.SORT_REMOVE);
    rulename2rule.put("SORT_INSTANCE", PruneEmptyRules.SORT_INSTANCE);
    rulename2rule.put("SORT_FETCH_ZERO_INSTANCE", PruneEmptyRules.SORT_FETCH_ZERO_INSTANCE);

    rulename2rule.put("UNION_MERGE", CoreRules.UNION_MERGE);
    rulename2rule.put("UNION_REMOVE", CoreRules.UNION_REMOVE);
    rulename2rule.put("UNION_TO_DISTINCT", CoreRules.UNION_TO_DISTINCT);
    rulename2rule.put("UNION_PULL_UP_CONSTANTS", CoreRules.UNION_PULL_UP_CONSTANTS);
    rulename2rule.put("UNION_INSTANCE", PruneEmptyRules.UNION_INSTANCE);
    rulename2rule.put("INTERSECT_INSTANCE", PruneEmptyRules.INTERSECT_INSTANCE);
    rulename2rule.put("MINUS_INSTANCE", PruneEmptyRules.MINUS_INSTANCE);

    //Config
    String path = System.getProperty("user.dir");
    String[] levels = path.split("/");
    // Use StringBuilder to rebuild the path without the last level
    StringBuilder modifiedPath = new StringBuilder();
    for(int i = 0; i < levels.length - 1; i++) {
        modifiedPath.append(levels[i]);
        // Add "/" between the levels but not at the end
        if(i < levels.length - 2) {
            modifiedPath.append("/");
        }
    }
    String newpath = modifiedPath.toString();

    JSONArray schemaJson = Utils.readJsonFile(newpath+"/data/data_llmr2/schemas/" + db_id + ".json");
// change to align python
//    JSONArray schemaJson = Utils.readJsonFile(path+"/main/schema.json");
    Rewriter rewriter = new Rewriter(schemaJson);

    //todo query formating
    String sql_input = input_sql;
//    String sql_input = "SELECT MAX(distinct l_orderkey) FROM lineitem where exists( SELECT MAX(c_custkey) FROM customer where c_custkey = l_orderkey GROUP BY c_custkey )";

    sql_input = sql_input.replace(";", "");
    RelToSqlConverter converter = new RelToSqlConverter(PostgresqlSqlDialect.DEFAULT);
//     SqlDialect dialect = SqlDialect.DatabaseProduct.SQLITE.getDialect();
//     RelToSqlConverter converter = new RelToSqlConverter(dialect);
    RelNode testRelNode = rewriter.SQL2RA(sql_input);


//    RelOptRule rule_instance = CoreRules.AGGREGATE_JOIN_REMOVE;
//     RelOptRule rule_instance = rulename2rule.get(input_rules.get(0));
//    RelOptRule rule_instance = rulename2rule.get("AGGREGATE_EXPAND_DISTINCT_AGGREGATES");
    RelNode new_node = testRelNode;
    RelNode rewrite_result = testRelNode;
    for(String rule:input_rules){
        RelOptRule rule_instance = rulename2rule.get(rule);
        HepProgramBuilder builder = getbuilder(rule_instance);
        HepPlanner hepPlanner = new HepPlanner(builder.addMatchOrder(HepMatchOrder.TOP_DOWN).build());
        for (int i = 0;i < 5;i++){
          hepPlanner.setRoot(new_node);
          rewrite_result = hepPlanner.findBestExp();
        }
//         hepPlanner.setRoot(new_node);
//         rewrite_result = hepPlanner.findBestExp();
        new_node = rewrite_result;
    }


//     HepPlanner hepPlanner = new HepPlanner(builder.addMatchOrder(HepMatchOrder.TOP_DOWN).build());
//     hepPlanner.setRoot(testRelNode);
//     RelNode rewrite_result = hepPlanner.findBestExp();
    System.out.println(testRelNode.explain());
    System.out.println(rewrite_result.explain());
    String rewrite_sql = converter.visitRoot(rewrite_result).asStatement().toSqlString(PostgresqlSqlDialect.DEFAULT).getSql();
//     String rewrite_sql = converter.visitRoot(rewrite_result).asStatement().toSqlString(dialect).getSql();
    if(rewrite_result.equals(testRelNode)){
      System.out.println("No changed!");
      rewrite_sql = sql_input;
    }
    System.out.println(sql_input);
    System.out.println(rewrite_sql);
    System.out.println(rewriter.getCostRecordFromRelNode(testRelNode));
    System.out.println(rewriter.getCostRecordFromRelNode(rewrite_result));
    EquivCheck.checkeq(rewriter, sql_input, rewrite_sql, testRelNode, rewrite_result);
  }
}




