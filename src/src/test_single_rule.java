import com.alibaba.fastjson.JSONArray;
import main.Rewriter;
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

import org.apache.calcite.adapter.java.ReflectiveSchema;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.externalize.RelWriterImpl;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlExplainLevel;

import java.io.PrintWriter;


import java.util.Arrays;
import java.util.List;

public class test_single_rule {
    public static HepProgramBuilder getbuilder(RelOptRule rule_instance){
        HepProgramBuilder builder = new HepProgramBuilder();
        // RelOptRule rule_instance = CoreRules.AGGREGATE_EXPAND_DISTINCT_AGGREGATES;
        builder.addRuleInstance(rule_instance);
        /*List<RelOptRule> t = Arrays.<RelOptRule>asList(CoreRules.AGGREGATE_PROJECT_MERGE,CoreRules.AGGREGATE_VALUES, PruneEmptyRules.AGGREGATE_INSTANCE);
        for(RelOptRule rule:t){
            builder.addRuleInstance(rule);
        }*/
        return builder;
    }
    public static void main(String[] args) throws Exception{
        String path = System.getProperty("user.dir");
        JSONArray schemaJson = Utils.readJsonFile(path+"/data/schemas_100000/tpch.json");
        Rewriter rewriter = new Rewriter(schemaJson);
        String testSql;
        testSql = "select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority from customer, orders, lineitem where c_mktsegment = 'MACHINERY' and c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate < date '1995-03-09' and l_shipdate > date '1995-03-09' group by l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate limit 10";
//        testSql = "SELECT T1.customer_name FROM customers AS T1 JOIN orders AS T2 ON T1.customer_id = T2.customer_id GROUP BY T1.customer_name , T1.customer_id  ORDER BY count(*) DESC LIMIT 1";
//        testSql = "SELECT DISTINCT COUNT ( t3.paperid ) FROM venue AS t4 JOIN paper AS t3 ON t4.venueid  =  t3.venueid JOIN writes AS t2 ON t2.paperid  =  t3.paperid JOIN author AS t1 ON t2.authorid  =  t1.authorid WHERE t1.authorname  =  'David M. Blei' AND t4.venuename  =  'AISTATS'";
//        testSql = "SELECT T1.name FROM browser AS T1 JOIN accelerator_compatible_browser AS T2 ON T1.id  =  T2.browser_id JOIN web_client_accelerator AS T3 ON T2.accelerator_id  =  T3.id WHERE T3.name  =  'CProxy' AND T2.compatible_since_calcite_year  >  1998";
//        testSql = "select l_discount,count (distinct l_orderkey), sum(distinct l_tax)\n" +
//                "from lineitem, part\n" +
//                "where l_discount > 100 group by l_discount;";
        /*testSql = "select distinct l_orderkey " +
                "from lineitem left join part on l_orderkey = p_partkey;";
        testSql = "SELECT\n" +
                "  MAX(distinct l_orderkey)\n" +
                "FROM\n" +
                "  lineitem\n" +
                "where\n" +
                "  exists(\n" +
                "    SELECT\n" +
                "      MAX(c_custkey)\n" +
                "    FROM\n" +
                "      customer\n" +
                "    where\n" +
                "      c_custkey = l_orderkey\n" +
                "    GROUP BY\n" +
                "      c_custkey\n" +
                "  );";
        */
        testSql = testSql.replace(";", "");
        RelNode testRelNode = rewriter.SQL2RA(testSql);
        RelToSqlConverter converter = new RelToSqlConverter(PostgresqlSqlDialect.DEFAULT);

        RelOptRule rule_instance = CoreRules.AGGREGATE_REMOVE;
//        ['FILTER_INTO_JOIN', 'JOIN_CONDITION_PUSH', 'AGGREGATE_REMOVE']

        HepProgramBuilder builder = getbuilder(rule_instance);
        HepPlanner hepPlanner = new HepPlanner(builder.addMatchOrder(HepMatchOrder.TOP_DOWN).build());
        hepPlanner.setRoot(testRelNode);
        RelNode rewrite_result = hepPlanner.findBestExp();

        final RelWriter relWriter = new RelWriterImpl(new PrintWriter(System.out), SqlExplainLevel.ALL_ATTRIBUTES, false);
        rewrite_result.explain(relWriter);

        System.out.println(testRelNode.explain());
        System.out.println(rewrite_result.explain());
        String rewrite_sql = converter.visitRoot(rewrite_result).asStatement().toSqlString(PostgresqlSqlDialect.DEFAULT).getSql();
        if(rewrite_result.equals(testRelNode)){
            System.out.println("No changed!");
            return;
        }
        System.out.println(testSql);
        System.out.println(rewrite_sql);
        System.out.println(rewriter.getCostRecordFromRelNode(rewrite_result));
    }
}
