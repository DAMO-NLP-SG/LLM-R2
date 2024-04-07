import com.alibaba.fastjson.JSONArray;
import com.google.gson.JsonObject;
import main.Node;
import main.Rewriter;
import main.Utils;

import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import verify.*;
//
// import verify.*;
import org.apache.calcite.plan.RelOptUtil;

public class test_eq {
  public static void main(String[] args) throws Exception {

    //DB Config
    String path = System.getProperty("user.dir");
    JSONArray schemaJson = Utils.readJsonFile(path+"/src/main/schema.json");
    Rewriter rewriter = new Rewriter(schemaJson);


    //todo query formating
    String testSql = testSql = "select * from orders where (o_orderpriority + o_orderkey > 10 and o_orderkey < 100+2) and (1999 + 1 < o_totalprice and o_orderpriority like 'abcd') ";

    testSql = "SELECT\n" +
            "  *\n" +
            "from\n" +
            "  customer\n" +
            "where\n" +
            "  c_custkey > (\n" +
            "    SELECT\n" +
            "      MAX(l_orderkey)\n" +
            "    FROM\n" +
            "      lineitem\n" +
            "    where\n" +
            "      c_custkey = l_partkey\n" +
            "  )" +
            "order by c_custkey;";
    testSql = "(select * from lineitem order by l_orderkey) union (select * from lineitem order by l_partkey)";
    testSql = "SELECT\n" +
            "  *\n" +
            "from\n" +
            "  customer\n" +
            "where\n" +
            "  c_custkey > (\n" +
            "    SELECT\n" +
            "      MAX(l_orderkey)\n" +
            "    FROM\n" +
            "      lineitem\n" +
            "    where\n" +
            "      c_custkey = l_partkey and True\n" +
            "  );";
    testSql = "select c.c_custkey" +
            " from customer as c" +
            " left join lineitem as l\n" +
            "   on c.c_custkey = l.l_orderkey";
    String sql2 = "select c.c_custkey" +
            "           from customer as c" +
            "   left join lineitem as l" +
            " on c.c_custkey = l.l_orderkey";
    // 使用方法：修改testSql与sql2。
    // testSql = "select distinct c1.c_custkey as ck from customer c1, customer c2, orders o where c1.c_custkey = c2.c_custkey and c1.c_custkey = o.o_orderkey";
    testSql = testSql.replace(";", "");
    sql2 = sql2.replace(";","");
    RelNode testRelNode = rewriter.SQL2RA(testSql);
    double origin_cost = rewriter.getCostRecordFromRelNode(testRelNode);
    Node resultNode = new Node(testSql,testRelNode, (float) origin_cost,rewriter, (float) 0.1,null,"original query");
    Node res = resultNode.UTCSEARCH(5, resultNode,1);
    // System.out.println(RelOptUtil.toString(rewriter.removeOrderbyNCalc(testRelNode,null,0)));
    System.out.println("--------Equality Check: Two Relnodes: -------------");
    RelNode testRelNode_v = rewriter.removeOrderbyNCalc(testRelNode,null,0);
    RelNode rewriteRelNode_v = rewriter.removeOrderbyNCalc(testRelNode,null,0);

    RelNode sql2node = rewriter.removeOrderbyNCalc(rewriter.SQL2RA(sql2),null,0);
    JsonObject eqres = verifyrelnode.verifyrelnode(testRelNode_v, sql2node, testSql, res.state);
    System.out.println("-------Equality Check Res: --------------");
    System.out.println(eqres);
    // EquivCheck.checkeq(rewriter, testSql, res.state, testRelNode, res.state_rel);
  }
}
