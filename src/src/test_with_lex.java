import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.google.gson.JsonObject;
import main.GenerateSchema;
import org.apache.calcite.adapter.jdbc.JdbcSchema;
import org.apache.calcite.config.Lex;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.rel2sql.RelToSqlConverter;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.util.SourceStringReader;
import org.apache.calcite.sql.dialect.*;
import javax.sql.DataSource;
import main.DBConn;
import main.Rewriter;
import main.Utils;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static org.apache.calcite.avatica.util.Casing.UNCHANGED;
import static org.apache.calcite.avatica.util.Quoting.DOUBLE_QUOTE;

import org.apache.commons.lang3.tuple.Pair;
import verify.verifyrelnode;
import com.google.gson.JsonObject;

public class test_with_lex {
    public static void main(String[] args) throws Exception {

        String testSql = "select * from (select row_.*, aol_rownum as total_rownum from (select distinct (aol.ol_id) olId, aol.ol_nbr olNbr, aol.so_date soDate, aol.rownum as aol_rownum, (select c.region_name from tab1 c where c.common_region_id = aol.order_region_id) regionName, (select c.region_name from tab1 c where c.common_region_id = aol.so_lan_id) areaName, (select cc.name from tab2 cc where cc.channel_id = aol.channel_id and rownum < 2) channelName, (select '|' || sn1.name from tab3 as sn1 where sn1.staff_id = aol.staff_id and rownum < 2) staffName, (select t.service_name from tab4 t where t.service_kind = aol.service_type) serviceName, (select so.remark from tab5 so where so.service_offer_id = aol.action_type_name) remark, aol.access_number accessNumber from tab6 aol where aol.order_region_id < 10000 and aol.so_date >= '2022-01-01 00:00:00' and aol.so_date <= '2022-01-04 00:00:00' and not exists (select orl.ol_id from ol_rule_list orl where orl.ol_id = aol.ol_id)) row_ where aol_rownum <= 40000) as table_alias where table_alias.total_rownum >= 0";
        Vector<Pair<String, Vector<Pair<String, String>>>>schema_all = null;
        testSql = "select\n"
                + "    o_orderpriority,\n"
                + "    count(*) as order_count\n"
                + "from\n"
                + "    orders\n"
                + "where\n"
                + "    o_orderdate >= date '1997-07-01'\n"
                + "    and o_orderdate < date '1997-07-01' + interval '3' month\n"
                + "    and exists (\n"
                + "        select\n"
                + "            *\n"
                + "        from\n"
                + "            lineitem\n"
                + "        where\n"
                + "            l_orderkey = o_orderkey\n"
                + "            and l_commitdate < l_receiptdate\n"
                + "    )\n"
                + "group by\n"
                + "    o_orderpriority\n"
                + "order by\n"
                + "    o_orderpriority";
        // testSql = "select * from orders where o_orderpriority = 1 + 2";
        //Parse
        SqlParser.Config parserConfig = SqlParser.config().withLex(Lex.MYSQL).withUnquotedCasing(UNCHANGED).withCaseSensitive(false).withQuoting(DOUBLE_QUOTE);
        SqlNode t = SqlParser.create(testSql, parserConfig).parseStmt();

        //validate
        String path = System.getProperty("user.dir");
        JSONArray jobj = Utils.readJsonFile(path+"/src/main/schema.json");
        SchemaPlus rootSchema = GenerateSchema.generate_schema(jobj, schema_all);
        FrameworkConfig planner_config = Frameworks.newConfigBuilder().defaultSchema(rootSchema).parserConfig(parserConfig).build();
        Planner planner = Frameworks.getPlanner(planner_config);

        planner.close();
        planner.reset();
        SqlNode sql_node = planner.parse(new SourceStringReader(testSql));
        System.out.println("--------parsed--------");
        System.out.println(sql_node);
        sql_node = planner.validate(sql_node);
        System.out.println("--------validated--------");
        System.out.println(sql_node);
        RelRoot rel_root = planner.rel(sql_node);
        RelNode rel_node = rel_root.project();

        RelNode rel_org = rel_node;

        System.out.println("--------RELNODE PLAIN--------");
        System.out.println(RelOptUtil.toString(rel_node));
        System.out.println("--------Rel2SQL--------");
        RelToSqlConverter converter = new RelToSqlConverter(PostgresqlSqlDialect.DEFAULT);
        String new_sql = converter.visitRoot(rel_node).asStatement().toSqlString(PostgresqlSqlDialect.DEFAULT).getSql();

        System.out.println(new_sql);

        Deque stack = new LinkedList();
        stack.add(rel_node);
        while (stack.size()>0) {
            RelNode node = (RelNode) stack.pop();
            for (int i = 0; i < node.getInputs().size(); i++) {
                stack.add(node.getInputs().get(i));
                RelNode child = node.getInput(i);
                if (child instanceof LogicalAggregate){
                    if(((LogicalAggregate) child).getAggCallList().size() == 0){
                        node.replaceInput(i,child.getInput(0));
                    }
                    else if (((LogicalAggregate) child).getAggCallList().get(0).toString().contains("SINGLE_VALUE")){
                        node.replaceInput(i,child.getInput(0));
                    }
                    else if (((LogicalAggregate) child).getAggCallList().get(0).toString().contains("MIN")){

                        RelDataType records = child.getRowType();
                        List column_names = records.getFieldNames();
                        List column_types = RelOptUtil.getFieldTypeList(records);
                        int flag =  ((LogicalAggregate) child).getAggCallList().get(0).getArgList().get(0);
                        if((((String) column_names.get(flag)).contains("$")) && ((column_types.get(flag).toString()).contains("BOOLEAN"))){
                            node.replaceInput(i,child.getInput(0));
                        }
                        RelNode tmp = child.getInput(0);

                    }

                }
            }

        }
        System.out.println("--------Cleaned_Rel--------");
        System.out.println(RelOptUtil.toString(rel_node));

        String cleaned_sql = converter.visitRoot(rel_node).asStatement().toSqlString(PostgresqlSqlDialect.DEFAULT).getSql();
        System.out.println("--------Cleaned_SQL--------");
        System.out.println(cleaned_sql);
/*
        System.out.println("--------Equality-------------");
        JsonObject jsres = verifyrelnode.verifyrelnode(rel_org, rel_node, testSql, cleaned_sql);
        System.out.println(jsres);
*/
    }

}