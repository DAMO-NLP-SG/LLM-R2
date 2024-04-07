package main;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRules;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepMatchOrder;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rel2sql.RelToSqlConverter;
import org.apache.calcite.rel.rules.AggregateExpandDistinctAggregatesRule;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.PruneEmptyRules;
import org.apache.calcite.sql.dialect.PostgresqlSqlDialect;
import org.apache.commons.lang3.tuple.Pair;

import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;



public class HepOpt {
  HepPlanner hepPlanner;
  RelToSqlConverter converter;
  // IF YOU CHANGE THE MAP rule2ruleset, THEN DON'T FORGET TO CHANGE THE FILE standard.txt IN ORDER!!!
  Map<String, List<RelOptRule>> rule2ruleset = Map.of(
          "rule_agg", Arrays.<RelOptRule>asList(CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES,CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES_TO_JOIN,CoreRules.AGGREGATE_JOIN_TRANSPOSE_EXTENDED,CoreRules.AGGREGATE_PROJECT_MERGE,CoreRules.AGGREGATE_ANY_PULL_UP_CONSTANTS,CoreRules.AGGREGATE_UNION_AGGREGATE,CoreRules.AGGREGATE_UNION_TRANSPOSE,CoreRules.AGGREGATE_VALUES, PruneEmptyRules.AGGREGATE_INSTANCE),
          "rule_filter",Arrays.<RelOptRule>asList(CoreRules.FILTER_AGGREGATE_TRANSPOSE,CoreRules.FILTER_CORRELATE,CoreRules.FILTER_INTO_JOIN,CoreRules.JOIN_CONDITION_PUSH,CoreRules.FILTER_MERGE,CoreRules.FILTER_MULTI_JOIN_MERGE, CoreRules.FILTER_PROJECT_TRANSPOSE,CoreRules.FILTER_SET_OP_TRANSPOSE,CoreRules.FILTER_TABLE_FUNCTION_TRANSPOSE,CoreRules.FILTER_SCAN,CoreRules.FILTER_REDUCE_EXPRESSIONS,CoreRules.PROJECT_REDUCE_EXPRESSIONS,PruneEmptyRules.FILTER_INSTANCE),
          "rule_join",Arrays.<RelOptRule>asList(CoreRules.JOIN_EXTRACT_FILTER,CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE,CoreRules.JOIN_PROJECT_LEFT_TRANSPOSE,CoreRules.JOIN_PROJECT_RIGHT_TRANSPOSE,CoreRules.JOIN_LEFT_UNION_TRANSPOSE,CoreRules.JOIN_RIGHT_UNION_TRANSPOSE,CoreRules.SEMI_JOIN_REMOVE,CoreRules.JOIN_REDUCE_EXPRESSIONS,PruneEmptyRules.JOIN_LEFT_INSTANCE,PruneEmptyRules.JOIN_RIGHT_INSTANCE),
          "rule_project",Arrays.<RelOptRule>asList(CoreRules.PROJECT_CALC_MERGE,CoreRules.PROJECT_CORRELATE_TRANSPOSE,CoreRules.PROJECT_MERGE,CoreRules.PROJECT_MULTI_JOIN_MERGE, CoreRules.PROJECT_REMOVE,CoreRules.PROJECT_TO_CALC,CoreRules.PROJECT_SUB_QUERY_TO_CORRELATE,CoreRules.PROJECT_REDUCE_EXPRESSIONS,PruneEmptyRules.PROJECT_INSTANCE),
          "rule_cal",Arrays.<RelOptRule>asList(CoreRules.CALC_MERGE,CoreRules.CALC_REMOVE),
          "rule_orderby",Arrays.<RelOptRule>asList(CoreRules.SORT_JOIN_TRANSPOSE,CoreRules.SORT_PROJECT_TRANSPOSE,CoreRules.SORT_UNION_TRANSPOSE,CoreRules.SORT_REMOVE_CONSTANT_KEYS,CoreRules.SORT_REMOVE,PruneEmptyRules.SORT_INSTANCE,PruneEmptyRules.SORT_FETCH_ZERO_INSTANCE),
          "rule_union",Arrays.<RelOptRule>asList(CoreRules.UNION_MERGE,CoreRules.UNION_REMOVE,CoreRules.UNION_TO_DISTINCT,CoreRules.UNION_PULL_UP_CONSTANTS,PruneEmptyRules.UNION_INSTANCE,PruneEmptyRules.INTERSECT_INSTANCE,PruneEmptyRules.MINUS_INSTANCE)
  );
  Map <String, List<Boolean>> select_rule2ruleset_bitmap;
  Map <String, Pair<String, Integer>> rule2classidx;

  //todo different rules
  public  HepOpt() throws IOException {
    HepProgramBuilder builder = new HepProgramBuilder();
    this.hepPlanner = new HepPlanner(builder.addMatchOrder(HepMatchOrder.TOP_DOWN).build());
    this.select_rule2ruleset_bitmap = new HashMap<>();
    this.rule2classidx = new HashMap<>();
    //todo different dialects
    for(Map.Entry<String, List<RelOptRule>> entry:rule2ruleset.entrySet()){ // Initialize select_rule2ruleset_bitmap.
      List<Boolean> select_bitmap = new ArrayList<>();
      for(int i = 0; i < entry.getValue().size(); ++ i){
        select_bitmap.add(Boolean.FALSE); // default : use no rules.
      }
      select_rule2ruleset_bitmap.put(entry.getKey(), select_bitmap);
    }
    initrule2classidx(); // Initialize rule2classidx;
    updateSelectBitmap();
    this.converter = new RelToSqlConverter(PostgresqlSqlDialect.DEFAULT);
  }
   public void initrule2classidx() throws IOException {
// change to align python
//     BufferedReader bufReader = new BufferedReader(new FileReader("../rules_for_selected/standard.txt"));
     BufferedReader bufReader = new BufferedReader(new FileReader("rules_for_selected/standard.txt"));
     List<String> list = new ArrayList<>();
     String line = null, cur_class = null;
     int idx = 0;
     while(!Objects.equals(line = bufReader.readLine(), null)){
       if(rule2ruleset.containsKey(line)) {
         idx = 0;
         cur_class = line;
       } else {
         rule2classidx.put(line,Pair.of(cur_class, idx));
         idx ++;
       }
     }
   }
  public Boolean SingleAddRule(String rule){
    if(rule2classidx.containsKey(rule)){
      Pair<String,Integer> class_idx = rule2classidx.get(rule);
      List<Boolean> bitmap = select_rule2ruleset_bitmap.get(class_idx.getLeft());
      bitmap.set(class_idx.getRight(), Boolean.TRUE);
      select_rule2ruleset_bitmap.put(class_idx.getLeft(), bitmap);
      return Boolean.TRUE;
    }
    return Boolean.FALSE;
  }
  public Boolean SingleRemoveRule(String rule){
    if(rule2classidx.containsKey(rule)){
      Pair<String,Integer> class_idx = rule2classidx.get(rule);
      List<Boolean> bitmap = select_rule2ruleset_bitmap.get(class_idx.getLeft());
      bitmap.set(class_idx.getRight(), Boolean.FALSE);
      select_rule2ruleset_bitmap.put(class_idx.getLeft(), bitmap);
      return Boolean.TRUE;
    }
    return Boolean.FALSE;
  }
  public void updateSelectBitmap() throws IOException {
    BufferedReader bufReader = new BufferedReader(new FileReader("rules_for_selected/user_selected_rules.txt"));
    // change to align python
//    BufferedReader bufReader = new BufferedReader(new FileReader("../rules_for_selected/user_selected_rules.txt"));
    String line = null;
    while(!Objects.equals(line = bufReader.readLine(), null)){
      if(SingleAddRule(line)){
//        System.out.println("\u001B[32m" + line + " was added" + " successfully." + "\u001B[0m");
      } else {
        System.out.println("\u001B[31m" + "Failed to add rule " + line + ": The rule is NOT in \"standard.txt\", plz check if your spelling is correct." + "\u001B[0m");
      }
    }
  }
  public void updateRule(String rule){
    HepProgramBuilder builder = new HepProgramBuilder();
    for(int i = 0; i < this.rule2ruleset.get(rule).size(); ++ i){
      RelOptRule rule_instance = rule2ruleset.get(rule).get(i);
      Boolean selected = select_rule2ruleset_bitmap.get(rule).get(i);
      if(selected){
        builder.addRuleInstance(rule_instance);
        System.out.println(rule_instance.toString() + " RuleInstance is selected");
      }
      else System.out.println(rule_instance.toString() + " RuleInstance is not selected");
    }
    this.hepPlanner = new HepPlanner(builder.addMatchOrder(HepMatchOrder.TOP_DOWN).build());
  }

  public List findBest(RelNode relNode){

    List res = new ArrayList();
    RelNode finalNode = relNode;
    for (int i = 0;i < 5;i++){
      this.hepPlanner.setRoot(finalNode);
      finalNode = this.hepPlanner.findBestExp();
    }

    res.add(finalNode);
    String new_sql = converter.visitRoot(finalNode).asStatement().toSqlString(PostgresqlSqlDialect.DEFAULT).getSql();
    res.add(new_sql);
    res.add(this.hepPlanner.getRules());
    System.out.println(res);
    return res;
  }
}