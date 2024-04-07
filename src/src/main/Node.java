package main;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.apache.commons.logging.impl.AvalonLogger;

import javax.swing.LayoutStyle;
import org.checkerframework.checker.units.qual.A;

import java.sql.SQLException;
import java.util.*;

public class Node {
  int visits = 1;
  public String state;
  public RelNode state_rel;
  public float reward;
  Rewriter rewriter;
  public List children = new ArrayList();
  Node parent;
  List rewrite_sequence = new ArrayList();
  int node_num = 1;

  public float origin_cost;
  float gamma;
  int selected = 0;
  Map activatedRules = new HashMap();
  String name= "";


  public Node(String sql,
              RelNode state_rel,
              float origin_cost,
              Rewriter rewriter,
              float gamma,
              Node parent,
              String name
              ) throws Exception {
    this.state = sql;
    this.state_rel = state_rel;
    this.reward = (float) (origin_cost - rewriter.getCostRecordFromRelNode(state_rel));
    this.rewriter = rewriter;
    this.parent = parent;
    this.origin_cost = origin_cost;
    this.gamma = gamma;
    this.name = name;

  }
  public void add_child(String csql,RelNode relNode,float origin_cost,Rewriter rewriter,Map activatedRules,String name)
          throws Exception {
    Node child = new Node(csql,relNode,origin_cost,rewriter,this.gamma,parent,name);
    child.activatedRules = activatedRules;
    this.children.add(child);
  }

  public boolean rule_check(RelNode relNode, Class clazz){
    if(clazz.isInstance(relNode)){
      return true;
    }
    List rel_list = relNode.getInputs();
    for(int i = 0;i<rel_list.size();i++){
      if (rule_check((RelNode) rel_list.get(i), clazz)){
        return true;
      }
    }
    return false;
  }
  public void node_children() throws Exception {
    //todo rule selection

    for(String rule: this.rewriter.rule2class.keySet()) {
      if (rule_check(this.state_rel, rewriter.rule2class.get(rule))){

        System.out.println("\u001B[35m" + rule+ " Module is selected " + "\u001B[0m");
        List res = this.rewriter.singleRewrite(this.state_rel, rule);
        Map activatedRules = new HashMap();
        String csql = (String) res.get(1);
        double new_cost = this.rewriter.getCostRecordFromRelNode((RelNode) res.get(0));
        if (new_cost == -1){
          return;
        }
        if (new_cost<=this.origin_cost){
          //todo rule selection
          //self.used_rule_num = self.used_rule_num + self.rewriter.rulenums[self.rewriter.related_rule_list[i]]
          System.out.println("new node added..");
          this.add_child(csql, (RelNode) res.get(0),this.origin_cost,this.rewriter,activatedRules,rule);
        }
      }
    }

  }

  private boolean is_terminal(){
    if (this.children.size()>0 && this.children != null){
      return  false;
    }
    return true;
  }

  public Node UTCSEARCH(int buget,Node root,int parallel_num)
          throws Exception {
    root.selected = 1;
    int is_rewritten = 0;
    root.node_children();
    for (int i = 0; i<root.children.size();i++){
      Node child = (Node) root.children.get(i);
      child.parent = root;
    }
    List front_list = new ArrayList();
    for (int i = 0;i<buget;i++){
      //todo parallel search
      if (parallel_num ==1 || parallel_num > root.node_num){
        Node front = TREEPOLICY(root);
        front_list.clear();
        front_list.add(front);
      }

      for(int j = 0;j<front_list.size();j++){
        Node front = (Node) front_list.get(j);
        front.selected = 1;
        front.node_children();
        //todo rule selection
        //root.used_rule_num = root.used_rule_num + front.used_rule_num
        for (int k = 0;k<front.children.size();k++){
          Node c = (Node) front.children.get(k);
          c.parent = front;
        }
        root.node_num += front.children.size();
        float reward = FINDBESTREWARD(front);
        BACKUP(front,reward);

      }
    }

    Node best_node = FINDBESTNODE(root);
    return best_node;
  }

  private float FINDBESTREWARD(Node node){
    float reward = -1;
    for (int i = 0;i<node.children.size();i++){
      Node c = (Node) node.children.get(i);
      if(c.reward>reward){
        reward = c.reward;
      }
    }
    return reward;
  }

  private Node BESTCHILD(Node node){
    float bestscore = ((Node) node.children.get(0)).reward;
    List bestchildren = new ArrayList();
    bestchildren.add(node.children.get(0));
    for (int i = 0;i< node.children.size();i++){
      Node c = (Node) node.children.get(i);
      float exploit = c.reward/c.visits;
      float explore = (float) Math.sqrt(2.0*Math.log(node.visits)/c.visits);
      float score = exploit + node.gamma*explore;
      if (score > bestscore){
        bestchildren.clear();
        bestchildren.add(c);
      }
      else if (score == bestscore){
        bestchildren.add(c);
      }
    }
    Random random = new Random();
    int i = random.nextInt(bestchildren.size());
    return (Node) bestchildren.get(i);
  }

  private Node TREEPOLICY(Node node){
    while (!node.is_terminal()){
      node = BESTCHILD(node);
    }

    return node;
  }

  private void BACKUP(Node node,float reward){
    while(node != null){
      node.visits +=1;
      if (reward>node.reward){
        node.reward = reward;
      }
      node = node.parent;
    }
  }

  private Node FINDBESTNODE(Node best_node){
    while (!best_node.is_terminal()){
      float bestscore = ((Node) best_node.children.get(0)).reward;
      List bestchildren = new ArrayList();
      bestchildren.add(best_node.children.get(0));
      for (int i = 0;i< best_node.children.size();i++){
        Node c = (Node) best_node.children.get(i);
        float score = c.reward;
        if (score > bestscore){
          bestchildren.clear();
          bestchildren.add(c);
          bestscore = score;
        }
        else if (score == bestscore){
          bestchildren.add(c);
        }

      }
      Random random = new Random();
      int i = random.nextInt(bestchildren.size());
      best_node = (Node) bestchildren.get(i);

    }
    return best_node;
  }
}
