import com.alibaba.fastjson.JSONArray;
import main.Node;
import main.Rewriter;
import main.Utils;
import org.apache.calcite.rel.RelNode;

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
import org.apache.calcite.sql.SqlDialect;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import java.util.Scanner;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.util.List;



public class test {
  public static void main(String[] args) throws Exception {

    //Config
    Scanner scanner = new Scanner(System.in);
    String inputs = scanner.nextLine();
    Gson gson = new Gson();
    Type type = new TypeToken<List<Object>>(){}.getType();
    List<Object> inputList = gson.fromJson(inputs, type);
    String db_id = (String) inputList.get(0);
    String testSql = (String) inputList.get(1);

    String path = System.getProperty("user.dir");
    JSONArray schemaJson = Utils.readJsonFile(path+"/data/schemas_100000/" + db_id + ".json");
//     RelToSqlConverter converter = new RelToSqlConverter(PostgresqlSqlDialect.DEFAULT);
//     SqlDialect dialect = SqlDialect.DatabaseProduct.SQLITE.getDialect();
//     RelToSqlConverter converter = new RelToSqlConverter(dialect);
    Rewriter rewriter = new Rewriter(schemaJson);

    //todo query formating
//     String testSql = "SELECT T1.name FROM browser AS T1 JOIN accelerator_compatible_browser AS T2 ON T1.id  =  T2.browser_id JOIN web_client_accelerator AS T3 ON T2.accelerator_id  =  T3.id WHERE T3.name  =  'CProxy' AND T2.compatible_since_year  >  1998";

    RelNode testRelNode = rewriter.SQL2RA(testSql);
    double origin_cost = rewriter.getCostRecordFromRelNode(testRelNode);

//     RelToSqlConverter converter = new RelToSqlConverter(PostgresqlSqlDialect.DEFAULT);

    Node resultNode = new Node(testSql,testRelNode, (float) origin_cost,rewriter, (float) 0.1,null,"original query");

    Node res = resultNode.UTCSEARCH(20, resultNode,1);

    System.out.println("Original node: ");
    System.out.println(Utils.generate_json(resultNode));
    System.out.println("Rewrite node: ");
    System.out.println(Utils.generate_json(res));

    System.out.println(testSql);
    String rewrite_sql = res.state;
    RelNode rewriteNode = rewriter.SQL2RA(rewrite_sql);
    if(rewriteNode.equals(testRelNode)){
//       System.out.println("No changed!");
      rewrite_sql = testSql;
    }
    System.out.println(rewrite_sql);
//     System.out.println("root:"+res.state);
    System.out.println("Original cost: "+origin_cost);
    System.out.println("Optimized cost: "+rewriter.getCostRecordFromRelNode(res.state_rel));
//     System.out.println(Utils.generate_json(resultNode));
  }
}


